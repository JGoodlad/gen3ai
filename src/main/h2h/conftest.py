"""Fixtures of the head-to-head tests: the emission self-check build (once per worker) and two seeded
PERTURBED-fresh production-architecture checkpoints (the Lane H gate's own recipe, ``rust_eval.parity``)."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def built():
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    return True


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory):
    """``(zip A, zip B)`` in ``run_h2h_test/`` beside their ``model_config.json`` — two different weight sets
    of ONE architecture (the id of a bare zip is ``<dir>:<stem>``)."""
    from agents.training.rust_eval import parity as PAR

    dst = tmp_path_factory.mktemp("h2h") / "run_h2h_test"
    with PAR.declared_torch_state(1):
        trainee, sentinels, _cfg = PAR.build_models(dst, n_sentinels=1)
    return trainee, sentinels[0]


@pytest.fixture(scope="session")
def foreign(tmp_path_factory):
    """A seeded PERTURBED-fresh checkpoint of ANOTHER architecture — the production surface with
    ``--token-encoding static`` (a structural toggle: another state-dict signature and forward fingerprint) — in its
    own ``run_h2h_foreign/`` beside its ``model_config.json``. The static-vs-legacy screen's own arm pair (the X5 A/B's
    ``fixed_mass`` x ``blob`` pair was this fixture until the X5 version break deleted ``blob``)."""
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import parity as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    dst = tmp_path_factory.mktemp("h2h_foreign") / "run_h2h_foreign"
    dst.mkdir()
    args = production_args()
    args.token_encoding = "static"
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(7, args=args)
        perturb_(model.policy, seed=2700, scale=PERTURB_SCALE)
        path = dst / "snapshot_000000007000.zip"
        model.save(str(path))
    (dst / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return str(path)


@pytest.fixture(scope="session")
def third(tmp_path_factory):
    """A seeded PERTURBED-fresh checkpoint of a THIRD architecture — the production surface with
    ``--op-believed-lean`` OFF (a forward toggle: the same state-dict signature, another forward fingerprint) — in its
    own ``run_h2h_third/``: an engine of the other two architectures cannot serve it. (Not ``--move-prior-fusion`` OFF
    any more: X5's T0 move mixture reads the fused prior, so production cannot build without it.)"""
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import parity as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    dst = tmp_path_factory.mktemp("h2h_third") / "run_h2h_third"
    dst.mkdir()
    args = production_args()
    args.op_believed_lean = False
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(9, args=args)
        perturb_(model.policy, seed=2900, scale=PERTURB_SCALE)
        path = dst / "snapshot_000000009000.zip"
        model.save(str(path))
    (dst / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return str(path)
