"""ActiveContextEncoder tests — LiveView API (boosts + full gen3 volatile set)."""

import numpy as np

from agents.observation.active_context import ActiveContextEncoder
from agents.observation.constants import ACTIVE_CONTEXT_DIM, BOOSTS_DIM
from agents.observation.gen3_effects import VOLATILE_DIM, VOLATILE_SLOTS
from agents.battle.live_view import LivePokemon


def _mon(boosts=None, volatiles=None):
    """Minimal LivePokemon carrying just the fields the encoder reads."""
    return LivePokemon(
        species="zapdos", active=True, fainted=False, revealed=True,
        hp_fraction=1.0, status=None, types=("electric", "flying"),
        moves=(), item=None, ability=None,
        boosts=boosts or {}, volatiles=volatiles or {},
    )


def test_dimension():
    assert ActiveContextEncoder().dimension == ACTIVE_CONTEXT_DIM == BOOSTS_DIM + VOLATILE_DIM


def test_empty_context_is_zeros():
    assert ActiveContextEncoder().encode(None).sum() == 0.0
    assert ActiveContextEncoder().encode(_mon()).sum() == 0.0


def test_boosts_encoding():
    vec = ActiveContextEncoder().encode(_mon(boosts={"atk": 2, "def": -1}))
    assert vec[0] == 2 / 6.0 and vec[1] == 0.0   # atk +2
    assert vec[2] == 0.0 and vec[3] == 1 / 6.0   # def -1


def test_volatiles_full_set():
    vec = ActiveContextEncoder().encode(_mon(volatiles={"leechseed": 0, "taunt": 0}))
    off = BOOSTS_DIM
    assert vec[off + VOLATILE_SLOTS.index("leechseed")] == 1.0
    assert vec[off + VOLATILE_SLOTS.index("taunt")] == 1.0


def test_confusion_volatile():
    vec = ActiveContextEncoder().encode(_mon(volatiles={"confusion": 0}))
    assert vec[BOOSTS_DIM + VOLATILE_SLOTS.index("confusion")] == 1.0


def test_perish_counter_normalised():
    vec = ActiveContextEncoder().encode(_mon(volatiles={"perish2": 0}))
    # perish2 -> (2+1)/4 = 0.75 in the collapsed perish slot
    assert vec[BOOSTS_DIM + VOLATILE_SLOTS.index("perish")] == 0.75


def test_recovers_previously_dropped_volatiles():
    """Disable/Encore/Destiny Bond/Curse/Yawn were silently dropped by the old 9-slot
    encoder — they must now encode."""
    for v in ("disable", "encore", "destinybond", "curse", "yawn"):
        vec = ActiveContextEncoder().encode(_mon(volatiles={v: 0}))
        assert vec[BOOSTS_DIM + VOLATILE_SLOTS.index(v)] == 1.0, v


def test_flashfire_volatile_encodes():
    """Flash Fire's ability-volatile (the gap the e2e fuzz caught) must encode."""
    vec = ActiveContextEncoder().encode(_mon(volatiles={"flashfire": 0}))
    assert vec[BOOSTS_DIM + VOLATILE_SLOTS.index("flashfire")] == 1.0


def test_describe_vector_roundtrip():
    enc = ActiveContextEncoder()
    vec = enc.encode(_mon(boosts={"spe": 2}, volatiles={"substitute": 0}))
    d = enc.describe_vector(vec)
    assert d["boosts"] == {"spe": "+2"}
    assert any("substitute" in s for s in d["volatiles"])
