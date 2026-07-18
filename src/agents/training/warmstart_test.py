"""Unit tests for the disagreement-gated consensus warm-start CORE math (gen3_exploiter_consensus_warmstart_v1).

Only the pure numpy target construction is tested here (no models/bridge) — the "aligned parts" logic:
consensus + pairwise-JS disagreement + the temperature gate that SHARPENS agreement / FLATTENS forks.
"""
import numpy as np
import pytest

from agents.training.warmstart import (
    pairwise_js_disagreement, build_consensus_target, action_entropy)


def _onehotish(peak, a=4, hi=0.94):
    """A near-deterministic distribution peaked at action `peak` (rest share the remainder)."""
    p = np.full(a, (1.0 - hi) / (a - 1))
    p[peak] = hi
    return p


def test_pairwise_js_identical_is_zero():
    """Teachers that agree everywhere → zero disagreement."""
    p = np.array([[_onehotish(0), _onehotish(1)]])          # [1,2,4]
    tp = np.concatenate([p, p, p], axis=0)                   # 3 IDENTICAL teachers [3,2,4]
    d = pairwise_js_disagreement(tp)
    assert d.shape == (2,)
    assert np.allclose(d, 0.0, atol=1e-9)


def test_pairwise_js_single_teacher_is_zero():
    """A single teacher has no one to disagree with → all zeros (no crash)."""
    tp = np.array([[_onehotish(0), _onehotish(2)]])          # [1,2,4]
    d = pairwise_js_disagreement(tp)
    assert d.shape == (2,) and np.allclose(d, 0.0)


def test_pairwise_js_divergent_is_positive():
    """Teachers peaked on DIFFERENT actions → positive disagreement, larger the more they diverge."""
    # state 0: both peak action 0 (agree). state 1: peak 0 vs peak 3 (disagree hard).
    t1 = np.stack([_onehotish(0), _onehotish(0)])
    t2 = np.stack([_onehotish(0), _onehotish(3)])
    d = pairwise_js_disagreement(np.stack([t1, t2]))
    assert d[0] == pytest.approx(0.0, abs=1e-9)              # agreed state
    assert d[1] > 0.3                                        # forked state — clearly positive


def test_target_rows_are_masked_legal_distributions():
    """Target rows sum to 1 over LEGAL actions and put exactly 0 mass on illegal ones."""
    tp = np.stack([np.stack([_onehotish(0), _onehotish(1)]),
                   np.stack([_onehotish(0), _onehotish(2)])])           # [2,2,4]
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)     # action 3 / 2,3 illegal
    target, gate, d = build_consensus_target(tp, mask, tmax=3.0)
    assert target.shape == (2, 4)
    assert np.allclose(target.sum(-1), 1.0, atol=1e-5)
    assert target[0, 3] == pytest.approx(0.0, abs=1e-6)                 # illegal masked out
    assert target[1, 2] == pytest.approx(0.0, abs=1e-6) and target[1, 3] == pytest.approx(0.0, abs=1e-6)
    assert gate.shape == (2,) and d.shape == (2,)


def test_gate_sharpens_agreement_flattens_disagreement():
    """The core property: where teachers AGREE the target is SHARP (low entropy); where they DISAGREE
    it is FLATTENED (higher entropy) — so a new exploiter inherits the consensus but forks freely."""
    # 4 states, increasing disagreement: 0 = full agree, 3 = hard fork.
    a = 4
    t1 = np.stack([_onehotish(0), _onehotish(0), _onehotish(0), _onehotish(0)])
    t2 = np.stack([_onehotish(0), _onehotish(0), _onehotish(1), _onehotish(3)])
    mask = np.ones((4, a), dtype=np.float32)
    target, gate, d = build_consensus_target(np.stack([t1, t2]), mask, tmax=4.0)
    ent = action_entropy(target)
    # the fully-agreed state is the sharpest; the hardest-fork state is the flattest.
    assert ent[0] == pytest.approx(ent.min(), abs=1e-6)
    assert ent[3] == pytest.approx(ent.max(), abs=1e-6)
    assert ent[3] > ent[0] + 0.2                                        # a real entropy rise on the fork
    assert gate[0] <= gate[3]                                           # gate tracks disagreement


def test_tmax_one_recovers_plain_consensus():
    """tmax=1 ⇒ T≡1 everywhere ⇒ the target is exactly the (mask-renormalized) consensus, no gating."""
    tp = np.stack([np.stack([_onehotish(0), _onehotish(1)]),
                   np.stack([_onehotish(2), _onehotish(3)])])
    mask = np.ones((2, 4), dtype=np.float32)
    target, gate, d = build_consensus_target(tp, mask, tmax=1.0)
    consensus = tp.mean(0)
    consensus = consensus / consensus.sum(-1, keepdims=True)
    assert np.allclose(target, consensus, atol=1e-5)


def test_tmax_below_one_rejected():
    """tmax < 1 would SHARPEN rather than flatten — a footgun, rejected."""
    tp = np.stack([np.stack([_onehotish(0)]), np.stack([_onehotish(1)])])
    with pytest.raises(ValueError):
        build_consensus_target(tp, np.ones((1, 4), np.float32), tmax=0.5)


# ---------------------------------------------------------------- device derivation (ai_v7_22 crash)

class _DeviceProbeModel:
    """Minimal model stub: params on a chosen device; records the obs device it is forwarded on."""

    def __init__(self, device):
        import torch as th

        class _Dist:
            pass

        class _Policy:
            def __init__(self, dev):
                self._p = th.nn.Parameter(th.zeros(1, device=dev))
                self.seen_devices = []

            def parameters(self):
                return iter([self._p])

            def get_distribution(self, ob):
                self.seen_devices.append(ob["observation"].device)
                d = _Dist()
                inner = _Dist()
                inner.logits = th.zeros(ob["observation"].shape[0], ob["action_mask"].shape[1],
                                        device=ob["observation"].device)
                d.distribution = inner
                return d

        self.policy = _Policy(device)


def _run_device_probe(device):
    from agents.training.warmstart import masked_action_probs
    m = _DeviceProbeModel(device)
    obs = np.zeros((5, 7), np.float32)
    mask = np.ones((5, 4), np.float32)
    probs = masked_action_probs(m, obs, mask)
    assert probs.shape == (5, 4)
    assert all(str(d).startswith(device) for d in m.policy.seen_devices)


def test_masked_action_probs_derives_device_from_model_cpu():
    """ai_v7_22 launch-crash regression: tensors must be built on the MODEL's device, not a
    caller-passed string — a cuda student + cpu-default teachers made any shared device wrong."""
    _run_device_probe("cpu")


def test_masked_action_probs_derives_device_from_model_cuda():
    import torch as th
    if not th.cuda.is_available():
        pytest.skip("no CUDA")
    _run_device_probe("cuda")
