"""The privileged true-opponent-team block builder (`gen3_value_true_team_v1`).

Scope note: these are the builder's OWN contracts — slot order, the absent-slot spelling, the
active flag, and never raising on the per-decision path. The claim that the block is the SAME
encoding the flat obs vector uses is not a claim a stub can settle, so it is proved against a real
bridge battle in `agents/training/true_team_channel_integration_test.py` instead.
"""
import numpy as np

from agents.observation.constants import (
    POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM, POKEMON_VECTOR_DIM, TEAM_SIZE,
)
from agents.observation.true_team import (
    TRUE_TEAM_KEY, TRUE_TEAM_SHAPE, build_true_team_block, empty_true_team_block,
)

SP = {"skarmory": 227, "blissey": 242, "claydol": 344, "tyranitar": 248}


class _Mon:
    def __init__(self, species):
        self.species = species


class _MarkerEncoder:
    """Writes the mon's species name length as a marker so a row is traceable to its mon.

    It stands in for `PokemonEncoder` only where the assertion is about WHICH mon landed in WHICH
    row — never about what the encoding contains.
    """
    def __init__(self, raise_on=()):
        self.raise_on = set(raise_on)
        self.calls = []

    def encode(self, mon, battle, is_own=False, live_mon=None, **kw):
        assert is_own is True, "the privileged block must encode the mons as OWN (fully known)"
        if mon.species in self.raise_on:
            raise RuntimeError("simulated per-mon encode failure")
        self.calls.append(mon.species)
        return np.full(POKEMON_VECTOR_DIM, float(SP[mon.species]), dtype=np.float32)


class _Battle:
    def __init__(self, active=None):
        self.active_pokemon = active

    def live_view(self):
        raise RuntimeError("no live view in this fixture")


def test_empty_block_is_the_encoders_own_absent_slot_spelling():
    block = empty_true_team_block()
    assert block.shape == TRUE_TEAM_SHAPE == (TEAM_SIZE, POKEMON_FULL_DIM)
    assert block.dtype == np.float32
    # species_known == 0 on every row IS how the encoder writes an absent slot, which is what lets
    # the consumer mask the row through the channel it already understands.
    assert not block.any()


def test_the_key_name_is_stable():
    # The name is a wire contract between Gen3Env, RLPlayer and the extractor's route; a rename
    # that missed one of the three would be a KeyError deep in a rollout.
    assert TRUE_TEAM_KEY == "opp_true_team"


def test_rows_are_filled_in_species_num_ascending_order():
    enc = _MarkerEncoder()
    team = [_Mon("tyranitar"), _Mon("skarmory"), _Mon("claydol"), _Mon("blissey")]
    block = build_true_team_block(team, _Battle(), enc, species_to_num=SP)
    assert enc.calls == ["skarmory", "blissey", "tyranitar", "claydol"]   # 227, 242, 248, 344
    assert [float(block[i, 0]) for i in range(4)] == [227.0, 242.0, 248.0, 344.0]
    # the two unused rows stay ABSENT, not merely zero-by-accident
    assert not block[4].any() and not block[5].any()


def test_an_unmapped_species_sorts_last_instead_of_raising():
    SP2 = dict(SP, mystery=None)
    del SP2["mystery"]
    team = [_Mon("mystery"), _Mon("skarmory")]

    class _E(_MarkerEncoder):
        def encode(self, mon, battle, is_own=False, live_mon=None, **kw):
            self.calls.append(mon.species)
            return np.zeros(POKEMON_VECTOR_DIM, dtype=np.float32)

    e = _E()
    build_true_team_block(team, _Battle(), e, species_to_num=SP2)
    assert e.calls == ["skarmory", "mystery"], e.calls


def test_a_team_larger_than_six_is_truncated_and_an_empty_team_is_all_absent():
    enc = _MarkerEncoder()
    big = [_Mon("skarmory")] * (TEAM_SIZE + 3)
    assert build_true_team_block(big, _Battle(), enc, species_to_num=SP).shape == TRUE_TEAM_SHAPE
    assert len(enc.calls) == TEAM_SIZE
    assert not build_true_team_block([], _Battle(), _MarkerEncoder(), species_to_num=SP).any()


def test_the_active_flag_rides_the_active_mons_row():
    enc = _MarkerEncoder()
    tyr = _Mon("tyranitar")
    team = [tyr, _Mon("skarmory")]
    block = build_true_team_block(team, _Battle(active=tyr), enc, species_to_num=SP)
    # skarmory (227) sorts first, tyranitar (248) second — the flag must follow the MON, not row 0.
    assert float(block[0, POKEMON_ACTIVE_OFFSET]) == 0.0
    assert float(block[1, POKEMON_ACTIVE_OFFSET]) == 1.0


def test_a_per_mon_encode_failure_leaves_that_row_absent_rather_than_killing_the_rollout():
    """This rides the per-decision emit path. A privileged LABEL channel that can raise is strictly
    worse than one that degrades to 'unknown' for one slot: the first ends the episode."""
    enc = _MarkerEncoder(raise_on={"skarmory"})
    block = build_true_team_block([_Mon("skarmory"), _Mon("blissey")], _Battle(), enc,
                                  species_to_num=SP)
    assert not block[0].any()                     # skarmory's row: absent
    assert float(block[1, 0]) == 242.0            # blissey still encoded
