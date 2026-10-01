"""K9(a) THE LEARNER GOLDEN as a routine test (``learner_golden`` module docs).

One eager fp32 ``train()`` at the production loss surface on the committed buffer must reproduce the
recorded post-update parameter bytes and every pinned loss EXACTLY, under this interpreter's torch
build. Unmarked (routine tier): CPU, one thread, ~4 s for the update. It NEVER records — a failure is
fixed by finding what changed the update, or, when the change is intended, by

    python -m agents.training.learner_golden record --reason "..."

under every torch build that has an entry (``gen3ai_stable`` and ``gen3ai_torch28`` today).
"""
from __future__ import annotations

import pytest

from agents.training import learner_golden as L


def test_one_update_reproduces_the_recorded_golden_exactly():
    diffs, now = L.check()
    assert not diffs, (
        "the learner golden MOVED — what one PPO update computes changed:\n  " + "\n  ".join(diffs)
        + "\nIf that is intended, re-record deliberately (module docs): "
          "python -m agents.training.learner_golden record --reason \"...\" under EVERY torch build.")
    # the production loss surface is live, not vacuously matched: every production term logged a loss
    for k in ("win_prob/loss", "belief/aux_loss", "belief/move_loss", "belief/movelatent_loss",
              "belief/spread_loss", "belief/natureev_loss", "belief/hptype_loss", "belief/item_loss",
              "opp_intent/alpha_loss", "opp_intent/beta_loss", "train/policy_gradient_loss",
              "train/entropy_loss"):
        assert k in now["losses"], f"{k} was not logged — a production term did not fold on the pinned buffer"


def test_the_golden_has_teeth_an_advantage_normalisation_flip_fails_it():
    """Change ONE detail of what the update computes (per-micro-batch advantage normalisation off) and
    the golden must name the losses and parameter groups that moved."""
    entry = L.load_golden()["entries"].get(L.torch_key())
    if entry is None:
        pytest.fail(f"no golden recorded for torch {L.torch_key()} (see test above)")
    model = L.build_learner()
    model.normalize_advantage = False
    diffs = L.diff(entry, L.compute(model))
    assert any(d.startswith("post_params_sha256") for d in diffs), diffs
    assert any("train/policy_gradient_loss" in d for d in diffs), diffs
    assert not any(d.startswith("the INIT moved") for d in diffs), "the perturbation must be in the update only"


def test_a_missing_torch_build_fails_it_never_records(monkeypatch):
    before = L.GOLDEN_PATH.read_bytes()
    monkeypatch.setattr(L, "torch_key", lambda: "0.0.0+never")
    with pytest.raises(L.LearnerGoldenError, match="no learner golden recorded for torch 0.0.0"):
        L.check()
    assert L.GOLDEN_PATH.read_bytes() == before


def test_record_requires_a_reason():
    with pytest.raises(SystemExit):
        L.main(["record"])
    with pytest.raises(SystemExit):
        L.record("   ")
