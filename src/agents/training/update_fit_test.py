"""gen3_update_fit_v1 — the dry first update at startup (`agents/training/update_fit.py`):

* the learner after a dry update is EXACTLY the learner before it: the next real `train()` lands on
  bit-identical parameters and logs identical losses (the learner golden's surface, CPU);
* the verdict is deterministic — a headroom one MiB short of the declared 1,024 is REFUSED, an OOM
  inside the dry update is REFUSED, both FATAL_CONFIG; at the bar it passes;
* the trainer runs the check on both startup paths, where the superseded lower-bound check was.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import agents.training.update_fit as UF


def _learner():
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


def _update(model, seed=True):
    from agents.training import learner_golden as LG
    if seed:
        np.random.seed(LG.UPDATE_SEED)
        torch.manual_seed(LG.UPDATE_SEED)
    model.train()
    losses = {k: v for k, v in model.logger.name_to_value.items()
              if LG._is_pinned_key(k) and isinstance(v, (int, float, np.floating))}
    return LG.params_sha256(model), losses


def test_a_dry_update_leaves_the_learner_bit_identical():
    from agents.training import learner_golden as LG
    ref = _update(_learner())
    m = _learner()
    before = (LG.params_sha256(m), sorted(vars(m)), m._n_updates,
              {k: v.clone() for k, v in m.policy.optimizer.state_dict()["state"].get(0, {}).items()
               if torch.is_tensor(v)})
    np.random.seed(LG.UPDATE_SEED)                       # the dry update must not consume the streams
    torch.manual_seed(LG.UPDATE_SEED)
    r = UF.dry_update(m)
    assert r.oom is None and r.rows == 64 and "golden" in r.source
    assert (LG.params_sha256(m), sorted(vars(m)), m._n_updates) == before[:3]
    for k, v in before[3].items():
        assert torch.equal(m.policy.optimizer.state_dict()["state"][0][k], v), k
    assert not m.logger.name_to_value                   # its records do not reach the real update's
    got = _update(m, seed=False)                        # the streams continue from BEFORE the dry update
    assert got[0] == ref[0]                             # the next real update: bit-identical weights
    assert got[1] == ref[1]                             # ... and the same logged losses


def test_the_dry_update_really_trains_on_its_fixture():
    """Teeth for the test above: inside the dry update the weights DO move (it is a real update)."""
    from agents.training import learner_golden as LG
    m = _learner()
    h0 = LG.params_sha256(m)
    seen = {}
    real_train = type(m).train

    def spy(self):
        real_train(self)
        seen["after"] = LG.params_sha256(self)
    m.train = spy.__get__(m)
    UF.dry_update(m)
    assert seen["after"] != h0 and LG.params_sha256(m) == h0


def _reading(headroom, oom=None):
    card = 10_000.0
    return UF.FitReading(demand_mib=card - headroom, peak_alloc_mib=8_000.0, reserved_mib=9_000.0,
                         device_free_mib=1_000.0, card_mib=card, headroom_mib=headroom,
                         floor_alloc_mib=1_900.0, seconds=4.0, rows=98_304, source="fixture", oom=oom)


def test_the_verdict_is_deterministic_at_the_declared_bar(tmp_path):
    from main.exit_codes import TrainExitCode, exit_code_for
    assert UF.FIT_HEADROOM_MIB == 1024.0
    line = UF.verdict(_reading(1024.0), "staged", str(tmp_path))
    assert "headroom 1,024 MiB" in line and (tmp_path / "update_fit.json").exists()
    with pytest.raises(UF.UpdateWontFit, match="does not fit") as ei:
        UF.verdict(_reading(1023.0), "resident")
    assert exit_code_for(ei.value) == int(TrainExitCode.FATAL_CONFIG)
    with pytest.raises(UF.UpdateWontFit, match="OUT OF MEMORY"):
        UF.verdict(_reading(5_000.0, oom="CUDA out of memory. Tried to allocate 62.00 MiB"), "resident")


def test_a_learner_with_no_logger_yet_runs_the_dry_update_and_still_has_none():
    """At startup sb3's `_logger` does not exist (`_setup_learn` makes it) — measured: the first real
    launch died there (AttributeError). The dry update brings its own and leaves none behind."""
    from agents.training import learner_golden as LG
    m = _learner()
    del m._logger
    h0 = LG.params_sha256(m)
    r = UF.dry_update(m)
    assert r.oom is None and "_logger" not in vars(m) and LG.params_sha256(m) == h0


def test_the_check_is_inert_off_cuda():
    assert UF.check_update_fits(_learner()) is None


def test_the_trainer_runs_the_check_on_both_startup_paths():
    from main.train import model_build
    src = inspect.getsource(model_build)
    assert src.count("_update_fit.check_update_fits(model, model_dir)") == 2
    assert "check_device_batch_fits" not in src


def test_the_dry_update_runs_the_behaviour_probe_as_the_real_update_does(monkeypatch):
    """Under the Rust core the first real update runs K9(b)'s pre-loop PROBE (an eager grad-mode
    forward); the dry update runs it too — on log-probs it computed itself, so a fatal-mode check
    passes — and restores the check, the row versions and the weights."""
    from agents.training import learner_golden as LG
    from agents.training.rust_rollout import consistency as C
    calls = []
    real = C.behaviour_probe

    def spy(model):
        calls.append(str(model.behaviour_check))
        out = real(model)
        assert out is not None and out["behaviour/max_abs_dlogp_current"] < 1e-4
        return out
    monkeypatch.setattr(C, "behaviour_probe", spy)
    m = _learner()
    m.behaviour_check = "fatal"
    h0 = LG.params_sha256(m)
    UF.dry_update(m)
    assert calls == ["warn"]
    assert m.behaviour_check == "fatal" and getattr(m, "_rust_row_versions", None) is None
    assert LG.params_sha256(m) == h0


def test_the_probe_releases_its_graph():
    """gen3_probe_releases_graph_v1: after the probe no policy stash holds an autograd graph."""
    import numpy as np
    from agents.training.rust_rollout import consistency as C
    m = _learner()
    m.behaviour_check = "warn"
    m._rust_row_versions = np.zeros((int(m.rollout_buffer.buffer_size), int(m.rollout_buffer.n_envs)), np.int64)
    m._rust_version = 0
    C.behaviour_probe(m)
    held = [(type(mod).__name__, k) for mod in m.policy.modules() for k, v in vars(mod).items()
            if isinstance(v, torch.Tensor) and v.grad_fn is not None]
    assert not held and getattr(m.policy, "_last_pi_distribution", None) is None


def test_the_probe_s_forward_is_bit_identical_without_its_saved_tensors():
    """gen3_probe_releases_graph_v1: dropping the saved tensors changes no value — the probe's log-probs
    (and V) are bit-identical to a plain grad-mode forward's."""
    from agents.training.rust_rollout import consistency as C
    from stable_baselines3.common.utils import obs_as_tensor
    m = _learner()
    buf = m.rollout_buffer
    obs = obs_as_tensor({k: v.reshape(-1, *v.shape[2:])[:16] for k, v in buf.observations.items()}, m.device)
    acts = torch.as_tensor(buf.actions.reshape(-1)[:16]).long()
    masks = torch.as_tensor(buf.action_masks.reshape(-1, buf.action_masks.shape[-1])[:16])
    m.policy.set_training_mode(True)
    with torch.enable_grad():
        v0, l0, e0 = m.policy.evaluate_actions(obs, acts, action_masks=masks)
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(C._drop_saved, C._never_unpacked):
        v1, l1, e1 = m.policy.evaluate_actions(obs, acts, action_masks=masks)
    assert torch.equal(l0, l1) and torch.equal(v0, v1) and torch.equal(e0, e1)
    with pytest.raises(RuntimeError, match="never backpropagated"):
        l1.sum().backward()
