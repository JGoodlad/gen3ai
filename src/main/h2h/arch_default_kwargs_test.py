"""ONE ENGINE GROUP for two checkpoints that differ only by an extractor kwarg RECORDED AT ITS DEFAULT (X5 look 3,
FINDING 1; ``agents.inference.service.slots.canonical_extractor_kwargs``).

A checkpoint written at a later commit records ``policy_readout: 'tower'`` and ``oracle_reveal: 'off'`` in its zip's
``policy_kwargs``; an older one of the same architecture lacks both. Here the "older" one is made from a perturbed-fresh
production checkpoint (``conftest``) by saving it again with those two keys removed from its saved
``features_extractor_kwargs`` — same weights, the identical extractor at load. ``declare_engine`` must put the pair in
ONE slot group (before canonicalisation it declared two, and a look-3 plan of three such architectures was refused).
CPU only, in-process (unmarked tier: two production builds, ~10 s); nothing is written outside ``tmp_path``; no
battle is played."""
from __future__ import annotations

import shutil

import pytest

from main.h2h import play as PL
from main.h2h.arch import _load_host, declare_engine

DEFAULTED = ("policy_readout", "oracle_reveal")


@pytest.fixture(scope="module")
def pair(checkpoints, tmp_path_factory):
    """``(new ref, old ref)``: the conftest's checkpoint A, and A re-saved without the two defaulted kwargs."""
    from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

    a, _b = checkpoints
    new = PL.resolve_player(a)
    m = load_checkpoint_strict(a, device="cpu", **historical_load_kwargs(a))
    fek = m.policy_kwargs["features_extractor_kwargs"]
    assert {k: fek.get(k) for k in DEFAULTED} == {"policy_readout": "tower", "oracle_reveal": "off"}, \
        "precondition: a HEAD checkpoint records both kwargs at their defaults"
    m.policy_kwargs = {**m.policy_kwargs,
                       "features_extractor_kwargs": {k: v for k, v in fek.items() if k not in DEFAULTED}}
    dst = tmp_path_factory.mktemp("h2h_old_kwargs") / "run_h2h_old_kwargs"
    dst.mkdir()
    path = dst / "snapshot_000000001000.zip"
    m.save(str(path))
    shutil.copy(new.config_path, dst / "model_config.json")
    old = PL.resolve_player(str(path))
    assert old.sha256 != new.sha256
    return new, old


def test_the_old_zip_really_lacks_the_defaulted_kwargs_and_loads_the_same_extractor(pair):
    new, old = pair
    mo, mn = _load_host(old), _load_host(new)
    assert not set(DEFAULTED) & set(mo.policy.features_extractor_kwargs)
    assert set(DEFAULTED) <= set(mn.policy.features_extractor_kwargs)
    fo, fn = mo.policy.features_extractor, mn.policy.features_extractor
    assert (fo.policy_readout, fo.oracle_reveal) == (fn.policy_readout, fn.oracle_reveal) == ("tower", "off")


@pytest.mark.parametrize("order", ["new_first", "old_first"])
def test_declare_engine_puts_the_default_only_pair_in_one_group(pair, order):
    """FAILS ON REVERT: the uncanonical fingerprint put the old checkpoint in a second slot group."""
    new, old = pair
    p, o = (new, old) if order == "new_first" else (old, new)
    decl = declare_engine([(p, o)], _load_host)
    assert len(decl.archs) == 1, decl.block()
    assert decl.members[new.sha256] == decl.members[old.sha256] == 0
    assert decl.roles == (("player", "opponent"),)
    assert decl.combos == ((0, 0),)
