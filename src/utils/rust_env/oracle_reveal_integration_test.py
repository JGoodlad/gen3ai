"""The ORACLE REVEAL through the real env core (``--oracle-reveal``; gen3_oracle_reveal_v1, v137).

The Rust side is gated in ``src/rust_env/tests/oracle_reveal_test.rs`` (the OFF digest pinned to the build
before the reveal, the `off` vs `species` differential over real battles). This is the PYTHON half over the
real self-check ``cdylib``:

* the spec's ``oracle_reveal`` key reaches the core, and the levels Python lists are the levels Rust parses
  (an unknown one is refused by Python AND by the core);
* at the FIRST decision the `off` row states one opponent species (the lead) and the `species` row states all
  six — in the observation's opponent block, which is what the shared trunk reads;
* the label columns the learner trains on survive it: under `species` every opponent species is stated, so
  ``belief_species`` is all PAD, and the belief aux loss on those REAL labels returns ``None`` (no believed
  slot, no target) with no NaN — "the blob head's species target is now visible, nothing may crash" — while
  the same loss on the `off` labels trains.
"""
import numpy as np
import pytest

from agents.observation.constants import OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET
from utils.rust_env import ffi
from utils.rust_env import protocol as P

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N = 4


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env.build import ensure_built

    ensure_built("selfcheck")
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _teams():
    from utils.ladder_corpus import teams

    return [t for t in teams("commit")[:8]]


def _first_decision(lib, level):
    spec = P.spec_json(n=N, threads=1, teams=_teams(), names=("oa", "ob"), turn_limit=250, refusal_budget=4,
                       bank_dir=None, labels=("belief",), oracle_reveal=level)
    core = ffi.FfiCore(spec, lib=lib)
    try:
        for i in range(N):
            core.cols["ep_team"][i, :] = [2 * i % 8, (2 * i + 1) % 8]
            core.cols["ep_seed"][i, :] = [i + 1, 2, 3, 4]
        core.dispatch("RESET")
        obs = np.array(core.cols["obs"], copy=True)
        labels = {k: np.array(core.cols[k], copy=True) for k in ("belief_species", "belief_moves", "known_moves")}
        need = np.array(core.cols["need"], copy=True)
        return obs, labels, need
    finally:
        core.close()


def _known(obs_row):
    return [float(obs_row[OFFSET_OPP_TEAM + j * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET]) for j in range(6)]


def test_the_levels_python_lists_are_the_levels_rust_parses(lib):
    assert P.ORACLE_REVEAL_LEVELS == ("off", "species")
    with pytest.raises(ValueError, match="oracle_reveal"):
        P.spec_json(n=1, threads=1, teams=["x"], names=("a", "b"), turn_limit=None, refusal_budget=0, bank_dir=None,
                    oracle_reveal="full")
    # the core refuses a level it does not know, by name (a hand-built spec past Python's own check)
    import json

    spec = json.loads(P.spec_json(n=1, threads=1, teams=_teams()[:2], names=("a", "b"), turn_limit=None,
                                  refusal_budget=0, bank_dir=None))
    spec["oracle_reveal"] = "full"
    with pytest.raises(Exception, match="oracle_reveal"):
        ffi.FfiCore(json.dumps(spec), lib=lib)


def test_the_first_decision_states_the_whole_opponent_team_under_species_and_the_lead_alone_under_off(lib):
    obs_off, _, need = _first_decision(lib, "off")
    obs_sp, _, need_sp = _first_decision(lib, "species")
    assert (need == need_sp).all() and need.any()
    for i in range(N):
        for s in range(2):
            if not need[i, s]:
                continue
            assert _known(obs_off[i, s]) == [1, 0, 0, 0, 0, 0], "off: the seen lead alone"
            assert _known(obs_sp[i, s]) == [1, 1, 1, 1, 1, 1], "species: every opponent species is stated at turn 1"
            # the lead's slot, and everything outside the opponent block's tail, is `off`'s
            a, b = obs_off[i, s], obs_sp[i, s]
            tail = slice(OFFSET_OPP_TEAM + POKEMON_FULL_DIM, OFFSET_OPP_TEAM + 6 * POKEMON_FULL_DIM)
            assert (a[:tail.start] == b[:tail.start]).all() and (a[tail.stop:] == b[tail.stop:]).all()
            assert (a[tail] == 0).all() and (b[tail] != 0).any()


def test_the_belief_loss_on_the_real_oracle_labels_is_a_clean_none_and_off_still_trains(lib):
    import torch

    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    loss = InstrumentedMaskablePPO._belief_aux_loss
    g = torch.Generator().manual_seed(0)
    bl = {"species": torch.randn(N, 6, 400, generator=g, requires_grad=True),
          "moves": torch.randn(N, 6, 400, generator=g, requires_grad=True)}
    for level, trains in (("species", False), ("off", True)):
        _, labels, need = _first_decision(lib, level)
        sp = torch.from_numpy(labels["belief_species"][:, 0]).long()       # p1's labels, (N, 6)
        mv = torch.from_numpy(labels["belief_moves"][:, 0]).long()         # (N, 6, 4)
        out = loss(bl, sp, mv)
        if trains:
            assert out is not None and torch.isfinite(out[0]), "off: five hidden mons per row are the targets"
        else:
            assert (sp == -1).all() and (mv == -1).all(), "species: nothing is hidden, nothing is a target"
            assert out is None, "an all-PAD minibatch is the loss's declared no-op, never a NaN"
            km = labels["known_moves"][:, 0]
            assert (km[:, :, 0] >= 1).all(), "every stated species (seen or not) has its true moves as the known-slot label"
