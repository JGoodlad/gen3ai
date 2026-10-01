"""The Lane H gate's shared fixture (its three files: parity_test / parity_fixed_test /
parity_sampled_test): the emission self-check build, once per module."""
import pytest


@pytest.fixture(scope="module")
def built():
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    return True
