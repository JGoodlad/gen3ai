"""gen3_dist_critic_v1 (Phase B) — the distributional head becomes the critic. Unit-level checks
of the value-source routing + the resume gate, without a full PPO build."""
import types

import pytest
import torch as th

from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config)


class _FakeHead:
    def __init__(self, atoms):
        self.atoms = atoms
    def mean(self, logits):
        return (th.softmax(logits, dim=-1) * self.atoms).sum(-1, keepdim=True)


def _fake_policy(value_from_dist, logits=None, scalar=7.0):
    """A bare object exposing exactly what _critic_value touches (no SB3 build)."""
    p = types.SimpleNamespace()
    p._value_from_dist = value_from_dist
    p.popart = None
    atoms = th.tensor([-1.0, 0.0, 1.0])
    fe = types.SimpleNamespace(value_dist_head=_FakeHead(atoms), last_value_dist_logits=logits)
    p.features_extractor = fe
    p.value_net = lambda latent: th.full((latent.shape[0], 1), scalar)
    # bind the real methods
    p._denorm = Gen3DualHeadMaskablePolicy._denorm.__get__(p)
    p._critic_value = Gen3DualHeadMaskablePolicy._critic_value.__get__(p)
    return p


def test_critic_value_uses_ez_under_phase_b():
    # logits favor the +1 atom → E[Z] ≈ +1, distinct from the scalar 7.0
    logits = th.tensor([[-5.0, -5.0, 5.0]])
    p = _fake_policy(True, logits=logits, scalar=7.0)
    v = p._critic_value(th.zeros(1, 4))
    assert abs(v.item() - 1.0) < 0.05, v.item()          # E[Z], not the scalar


def test_critic_value_uses_scalar_when_off():
    p = _fake_policy(False, logits=th.zeros(1, 3), scalar=7.0)
    v = p._critic_value(th.zeros(1, 4))
    assert abs(v.item() - 7.0) < 1e-6                     # the scalar value_net


def test_phase_b_falls_back_to_scalar_if_logits_missing():
    """Defensive: value_from_dist ON but the head hasn't stashed logits → scalar, never a crash."""
    p = _fake_policy(True, logits=None, scalar=7.0)
    v = p._critic_value(th.zeros(1, 4))
    assert abs(v.item() - 7.0) < 1e-6


def test_value_from_dist_migrates_and_gates():
    d = _migrate_config({"config_version": 44})
    # The migration chain always advances to the LATEST version, not just to v45 — assert against
    # MODEL_CONFIG_VERSION so a later bump doesn't false-fail this v45 default check.
    assert d["config_version"] == MODEL_CONFIG_VERSION and d["value_from_dist"] is False
    assert "value_from_dist" in ModelVersion.__dataclass_fields__


def test_check_value_from_dist_resume_gate(capsys):
    # build a minimal version via migrate → construct
    base = _migrate_config({"config_version": 44})
    # only need the field for the check; use a lightweight object with the method bound
    v = types.SimpleNamespace(value_from_dist=False)
    v.check_value_from_dist = ModelVersion.check_value_from_dist.__get__(v)
    with pytest.raises(ModelVersionError):
        v.check_value_from_dist(True)                      # drift is FATAL
    v.check_value_from_dist(True, allow_change=True)       # migration hatch: no raise
    assert "MIGRATION" in capsys.readouterr().out
    v.check_value_from_dist(False)                         # match: no raise


def test_set_value_from_dist_runtime_toggle(capsys):
    """The runtime setter (the migration fix) flips _value_from_dist on the LIVE policy — SB3 load
    rebuilds from saved kwargs, so without this post-load call a first Phase-B resume is a silent
    no-op (grad/value_dist_share stays ~aux-level instead of ~0.5)."""
    p = types.SimpleNamespace()
    p._value_from_dist = False
    p.set_value_from_dist = Gen3DualHeadMaskablePolicy.set_value_from_dist.__get__(p)
    p.set_value_from_dist(True)
    assert p._value_from_dist is True
    assert "APPLIED at runtime -> True" in capsys.readouterr().out
    p.set_value_from_dist(True)  # no-op (unchanged) — no second notice
    assert capsys.readouterr().out == ""
