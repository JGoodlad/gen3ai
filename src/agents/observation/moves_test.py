import numpy as np
import pytest
from agents.observation.moves import MovesEncoder, HIDDEN_POWER_MOVE_NUM
from agents.observation.types import TypeEncoder
from poke_env.battle.move_category import MoveCategory


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeMove:
    """A revealed move on a mon. Note: the encoder now reads num/power/type/secondary/
    recoil/accuracy/never-miss from the gen3_movedex (keyed by id), NOT from these
    attributes — so ``base_power``/``move_type`` here are deliberately ignored for known
    moves. Category and PP are still read off the object."""

    def __init__(self, move_id, base_power=50, move_type="GRASS",
                 category=MoveCategory.SPECIAL, current_pp=10, max_pp=10,
                 accuracy=100):
        self.id = move_id
        self.base_power = base_power
        self._type = move_type
        self.category = category
        self.current_pp = current_pp
        self.max_pp = max_pp
        self.accuracy = accuracy

    @property
    def type(self):
        class _T:
            def __init__(self, n): self.name = n
            def __str__(self): return self.name
        return _T(self._type)


class FakeMon:
    def __init__(self, moves):
        self._moves = moves

    @property
    def moves(self):
        return {m.id: m for m in self._moves}


class FakeBattle:
    pass


# ---------------------------------------------------------------------------
# MovesEncoder — reference data comes from gen3_movedex (the real gen3 data)
# ---------------------------------------------------------------------------

@pytest.fixture
def reverse_mapping():
    # Real gen3 move nums for describe_vector's num→name lookup.
    return {33: "tackle", 85: "thunderbolt", 237: "hiddenpower", 53: "flamethrower"}


@pytest.fixture
def encoder(reverse_mapping):
    return MovesEncoder(reverse_mapping)


def test_unrecognized_move_raises(encoder):
    mon = FakeMon([FakeMove("notarealmove")])
    with pytest.raises(ValueError):
        encoder.encode(mon, FakeBattle())


def test_encode_known_move_enriches_from_dex(encoder):
    # thunderbolt: real gen3 data → num 85, 95 BP, has secondary
    vec = encoder.encode(FakeMon([FakeMove("thunderbolt")]), FakeBattle())
    assert vec[0] == 85.0                      # num
    assert vec[1] == pytest.approx(95 / 200.0)  # base power (dex, not the fake's 50)
    assert vec[2] == 1.0                        # has_secondary
    assert vec[6] == 1.0                        # known


def test_pp_is_normalized(encoder):
    vec = encoder.encode(FakeMon([FakeMove("tackle", current_pp=5, max_pp=35)]), FakeBattle())
    assert vec[7] == pytest.approx(5 / 64.0)
    assert vec[8] == pytest.approx(35 / 64.0)


def test_hidden_power_is_detected(encoder):
    vec = encoder.encode(FakeMon([FakeMove("hiddenpower")]), FakeBattle())
    assert vec[0] == float(HIDDEN_POWER_MOVE_NUM)
    assert vec[1] == pytest.approx(70 / 200.0)  # HP override power
    assert vec[4] == 0.0                          # HP type unknown → Normal=0


def test_known_flag_set_for_revealed_moves(encoder):
    vec = encoder.encode(FakeMon([FakeMove("tackle")]), FakeBattle())
    assert vec[6] == 1.0


def test_describe_vector_lists_moves(encoder):
    vec = encoder.encode(FakeMon([FakeMove("tackle"), FakeMove("thunderbolt")]), FakeBattle())
    desc = encoder.describe_vector(vec)
    assert "tackle" in desc["moves"]
    assert "thunderbolt" in desc["moves"]


def test_get_sorted_moves_is_deterministic(encoder):
    v1 = encoder.encode(FakeMon([FakeMove("tackle"), FakeMove("thunderbolt")]), FakeBattle())
    v2 = encoder.encode(FakeMon([FakeMove("thunderbolt"), FakeMove("tackle")]), FakeBattle())
    assert np.array_equal(v1, v2)


def test_type_slot_is_dex_driven(encoder):
    vec = encoder.encode(FakeMon([FakeMove("flamethrower")]), FakeBattle())
    assert vec[4] == float(TypeEncoder.TYPE_TO_IDX["FIRE"])
    assert vec[4] != float(TypeEncoder.TYPE_TO_IDX["GRASS"])


def test_dex_type_wins_over_move_object_type(encoder):
    # The fake claims type "WRONG"; the dex's real type (Electric) must win.
    vec = encoder.encode(FakeMon([FakeMove("thunderbolt", move_type="WRONG")]), FakeBattle())
    assert vec[4] == float(TypeEncoder.TYPE_TO_IDX["ELECTRIC"])


def test_category_slot_from_move_object(encoder):
    # Category is still read off the live move object, not the dex.
    vec = encoder.encode(
        FakeMon([FakeMove("thunderbolt", category=MoveCategory.SPECIAL)]), FakeBattle()
    )
    assert vec[5] == 2.0  # special
