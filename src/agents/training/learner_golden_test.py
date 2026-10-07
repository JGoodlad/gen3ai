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
    # (X5: the hidden-team belief is the set BCE, the intent is the flat pointer — `belief/aux_loss` and
    # `opp_intent/alpha_loss` / `beta_loss` were the deleted blob path's)
    for k in ("win_prob/loss", "belief/set_aux_loss", "belief/move_loss", "belief/movelatent_loss",
              "belief/spread_loss", "belief/natureev_loss", "belief/hptype_loss", "belief/item_loss",
              "opp_intent/flat_loss", "train/policy_gradient_loss", "train/entropy_loss"):
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


def test_the_default_slot_is_the_fixed_mass_entry_moved_verbatim():
    """The X5 version break MOVED the fixed_mass arm's committed entry into the default slot without
    re-recording it: the init / post hashes are the pre-break arm's (`version_break_identity_2026-10-07`'s
    reference, captured at 26131c0c), the buffer is the arm's seed-18 rollout and the blob slot is gone.
    FAILS if anyone re-records the golden without a new reason row, or restores an arm block."""
    g = L.load_golden()
    e = g["entries"]["2.8.0+cu126"]
    assert e["init_params_sha256"] == "47c4c5fdde7b4dc171db6f758eda0da097ab2b44224e61fc44fd67a736d92f69"
    assert e["post_params_sha256"] == "70a26bfc2cfb2ed37f6a951ce791e4c49271c0f4bce84a4a68f96f2e8d977750"
    assert g["buffer_sha256"] == "4b48eaf0a08c0382b655c79bc8e27aea5c3980f7594856479ab1056a08c66539"
    assert g["perturb"] == "name_keyed" and "arms" not in g
    assert L.RECORD_RUN_SEED == 18
    assert "MOVED VERBATIM" in g["history"][-1]["reason"]


def test_record_requires_a_reason():
    with pytest.raises(SystemExit):
        L.main(["record"])
    with pytest.raises(SystemExit):
        L.record("   ")


def test_the_golden_recipe_is_READ_from_the_production_recipe_and_matches_the_recorded_one():
    """The golden's recipe knobs come from K10(a)'s block (and sb3's defaults for the knobs no launch
    sets), never a hand copy; only the shape + the documented LR are overrides. A drift — the
    production recipe moved, or someone hand-edited a value back in — fails HERE, naming the knob,
    before the parameter hashes do."""
    import inspect

    from sb3_contrib import MaskablePPO

    from main.train.recipe_surface import production_recipe
    r = L.golden_recipe()
    prod = production_recipe()
    for k in L._FROM_RECIPE:
        assert r[k] == prod[k], (k, r[k], prod[k])
    sig = inspect.signature(MaskablePPO.__init__).parameters
    for k in L._FROM_SB3_DEFAULTS:
        assert r[k] == sig[k].default, k
    assert set(L.GOLDEN_OVERRIDES) == {"n_epochs", "batch_size", "grad_accum_steps", "learning_rate"}
    recorded = L.load_golden().get("recipe")
    assert recorded == r, ("the recorded golden was computed under a different recipe — re-record "
                           f"deliberately: {recorded} vs {r}")
