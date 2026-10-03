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

from agents.model.compile_trainer import (_MAX_NUMERIC_DRIFT, CompileTrainerError, check_numerics,
                                          check_shape_stability, preflight_compile_trainer,
                                          resolve_device)


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


def test_the_fp32_rule_is_one_bar_on_max_compiled_minus_eager_and_a_nan_fails():
    """fp32 'highest' is the ONLY precision (TF32 retired, K2): the rule is 1e-4 on max|compiled -
    eager| and there is no second, relative rule and no reference arm to consult."""
    assert _MAX_NUMERIC_DRIFT == 1e-4
    assert "fp32" in check_numerics(9e-5) and "< 0.0001" in check_numerics(9e-5)
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        check_numerics(2e-4)
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        check_numerics(float("nan"))                      # NaN FAILS, never sails through
    import inspect
    assert set(inspect.signature(check_numerics).parameters) == {"err", "what", "tol"}


def test_no_tf32_machinery_remains_in_the_compile_gate():
    """The retired TF32 rule's names must be gone, not merely unused: a precision argument or a
    reference arm left behind would let a TF32 request through the gate again."""
    import inspect

    from agents.model import compile_trainer as ct
    for gone in ("_TF32_K", "_TF32_EPS", "fp32_reference", "parity_verdict", "_matmul_precision"):
        assert not hasattr(ct, gone), gone
    for fn in (ct.decision_verdicts, ct.train_verdict):
        assert not {"precision", "reference", "eager_err"} & set(inspect.signature(fn).parameters), fn


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
    """The declared region is the ONLY compiled learner surface (the torch-2.5.1 extractor-only
    compile was deleted, K1 2026-10-02; the rollout region R0, P10-E): a learner without `_micro_static` cannot build R1, so it is
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
        arm_compile_sentinel(m, batch_size=16)
    lines = []
    m._micro_static = lambda *a: None
    preflight_compile_trainer(m, True, emit=lines.append)
    assert lines and "DECLARED REGION (R1" in lines[0]


def test_resolve_device_reads_the_models_real_device():
    assert resolve_device(_FakeFE("cpu")).type == "cpu"


# --------------------------------------------------------------- shape stability (the real hazard)


def _stable(**kw):
    base = dict(update_rows=2048 * 48, batch_size=4096)   # gen-10's config: 98,304 rows per update
    base.update(kw)
    return base


def test_the_production_config_is_accepted():
    """gen-10: 2048*48 = 98304 = 24 x 4096 exactly — every micro-batch is full."""
    check_shape_stability(**_stable())


def test_a_remainder_microbatch_is_refused_with_a_concrete_suggestion():
    """A remainder is a SECOND micro-batch shape replayed every epoch — one the single compiled R1
    graph does not declare. The error must not leave the reader doing arithmetic — it names a batch
    size that actually divides."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(batch_size=5000))
    msg = str(e.value)
    assert "remainder" in msg
    import re
    m = re.search(r"e\.g\. (\d+)", msg)
    assert m, f"no concrete suggestion in: {msg}"
    suggested = int(m.group(1))
    assert (2048 * 48) % suggested == 0, f"suggested {suggested} does not divide the update"
    assert suggested <= 5000


def test_the_suggestion_is_the_LARGEST_divisor_that_fits():
    """Suggesting 1 would be technically correct and useless."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(batch_size=5000))
    import re
    assert int(re.search(r"e\.g\. (\d+)", str(e.value)).group(1)) == 4096


def test_a_zero_batch_size_does_not_divide_by_zero():
    check_shape_stability(**_stable(batch_size=0))     # unknown -> not our call to police


def test_the_stability_check_judges_the_REAL_update_size_not_n_steps_times_n_envs():
    """F9 (P10-E): the check took `n_steps * n_envs`, but the update is `--rollout-target-samples` rows
    (the production recipe sets it independently), so a VALID run (target divides, product does not)
    was refused / had `--compile-trainer` quietly switched off, and an invalid one (product divides,
    target does not) passed. It now takes the update size. Fails on revert: the signature goes back to
    (n_steps, n_envs) and the callers' real-size math is lost."""
    import inspect
    assert set(inspect.signature(check_shape_stability).parameters) == {"update_rows", "batch_size"}
    from main.train.compile_flags import update_rows_for
    # a valid run whose n_steps*n_envs (1000*48 = 48,000) does NOT divide by 4096 but whose target does
    valid = update_rows_for(rollout_target_samples=98_304, n_steps=1000, n_envs=48)
    assert valid == 98_304 and 1000 * 48 % 4096 != 0
    check_shape_stability(update_rows=valid, batch_size=4096)
    # an invalid one: n_steps*n_envs divides (98,304) but the target (100,000) does not
    bad = update_rows_for(rollout_target_samples=100_000, n_steps=2048, n_envs=48)
    assert bad == 100_000 and 2048 * 48 % 4096 == 0
    with pytest.raises(CompileTrainerError, match=r"100,000 rows"):
        check_shape_stability(update_rows=bad, batch_size=4096)


# --------------------------------------------- the per-parameter bar has NO default (P10-E, item 1)


def _arms_with_sizes(eps=1e-5, n=6):
    g = torch.Generator().manual_seed(3)
    sizes = [50] * n
    eager = torch.randn(sum(sizes), generator=g)
    comp = eager * (1.0 + eps)
    f = torch.randn(2, 4, generator=torch.Generator().manual_seed(2))
    mk = lambda gr: {"features": f, "grad": gr, "grad_sizes": torch.tensor(sizes)}  # noqa: E731
    return mk(eager), mk(comp), [f"p{i}" for i in range(n)]


def test_the_per_parameter_rule_REFUSES_a_caller_that_names_no_bar():
    """`train_verdict`'s default `param_bar` was the retired extractor gate's TF32-era 1e-3 — a bar that
    flags R1's healthy fp32 noise (2.47e-3 on arm C's weights) as a miscompile, picked silently by any
    caller that forgot its own. The rule now needs the caller's bar whenever the arms carry per-parameter
    sizes, and says so; the bar a caller names is the bar applied. Fails on revert (a default returns, so
    a healthy 1e-5 reading would pass with no bar named)."""
    e, c, names = _arms_with_sizes(1e-5)
    from agents.model.compile_trainer import train_verdict
    with pytest.raises(CompileTrainerError, match="needs the caller's own bar"):
        train_verdict(eager=e, compiled=c, allow_vacuous=True, param_names=names)
    line = train_verdict(eager=e, compiled=c, allow_vacuous=True, param_names=names, param_bar=1e-2)
    assert "<= 0.01" in line
    e, c, names = _arms_with_sizes(5e-2)
    with pytest.raises(CompileTrainerError, match="DISAGREES with eager on"):
        train_verdict(eager=e, compiled=c, allow_vacuous=True, param_names=names, param_bar=1e-2)


def test_the_retired_extractor_gate_bars_are_gone():
    from agents.model import compile_trainer as ct
    assert not hasattr(ct, "_MAX_PARAM_GRAD_REL") and not hasattr(ct, "_MAX_PARAM_GRAD_REL_TRAINED")


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
    ok = decision_verdicts(eager=base, compiled={k: v.clone() for k, v in base.items()})
    assert len(ok) == 3
    bad_v = {k: v.clone() for k, v in base.items()}
    bad_v["value"] += 2 * _FP32_TOL["value"]
    with pytest.raises(CompileTrainerError, match="value"):
        decision_verdicts(eager=base, compiled=bad_v)
    bad_p = {k: v.clone() for k, v in base.items()}
    bad_p["legal_logprob"][0, 3] -= 2 * _FP32_TOL["legal_logprob"]
    with pytest.raises(CompileTrainerError, match="legal_logprob"):
        decision_verdicts(eager=base, compiled=bad_p)


def test_train_graph_gradient_cosine_is_gated():
    """The single-graph miscompile's gradient had cosine 0.778 to eager's — PPO stepping the wrong
    way. The train verdict must refuse that."""
    from agents.model.compile_trainer import _MIN_GRAD_COSINE, train_verdict
    g = torch.randn(1000, generator=torch.Generator().manual_seed(0))
    f = torch.randn(2, 4, generator=torch.Generator().manual_seed(2))   # informative rows
    assert "grad cosine" in train_verdict(eager={"features": f, "grad": g},
                                          compiled={"features": f, "grad": g.clone()})
    skew = g + 0.6 * torch.randn(1000, generator=torch.Generator().manual_seed(1))
    with pytest.raises(CompileTrainerError, match="gradient DISAGREES"):
        train_verdict(eager={"features": f, "grad": g}, compiled={"features": f, "grad": skew})
    assert _MIN_GRAD_COSINE == 0.9999
