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
