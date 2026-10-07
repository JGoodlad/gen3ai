"""The FORWARD FINGERPRINT's canonical extractor kwargs (``slots.canonical_extractor_kwargs``; X5 look 3, FINDING 1).

A checkpoint written at a later commit RECORDS extractor kwargs at their defaults (``policy_readout: 'tower'``,
``oracle_reveal: 'off'``) that an older checkpoint of the same architecture lacks. Both build the same extractor, so
they must share a fingerprint (and so one T2 slot group, one ``main.h2h`` engine group); any NON-default difference must
still split, including a recorded value that is not what an ABSENT key means at load. "What absent means" is the
extractor class's constructor-signature default — the flag registry's CLI default is not it
(``attend_unrevealed_opponents``: True in the registry, False in the signature).

Synthetic policies at the fingerprint level: ``_get_constructor_parameters`` is the one thing the fingerprint reads.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from agents.inference.service.slots import canonical_extractor_kwargs, forward_fingerprint
from agents.model.features_extractor import Gen3FeaturesExtractor


class _Ext:
    """A synthetic extractor class: only its constructor SIGNATURE is ever read."""

    def __init__(self, observation_space: Any, mode: str = "off", width: int = 0, flag: bool = False,
                 frac: float = 0.0, cats: tuple = ()):
        raise AssertionError("the fingerprint must never construct the extractor")


class _Policy:
    def __init__(self, fek: Dict[str, Any], cls: Any = _Ext, critic: str = "winprob"):
        self._params = {"net_arch": {"pi": [], "vf": []}, "features_extractor_class": cls,
                        "features_extractor_kwargs": dict(fek)}
        self._critic_mode = critic

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(self._params)


def fp(fek: Dict[str, Any], cls: Any = _Ext) -> str:
    return forward_fingerprint(_Policy(fek, cls))


# --- the synthetic class -----------------------------------------------------------------------------------------

def test_a_kwarg_recorded_at_its_default_shares_the_fingerprint_of_one_absent():
    """FAILS ON REVERT: before canonicalisation every recorded key entered the hash."""
    assert fp({"width": 4}) == fp({"width": 4, "mode": "off"}) == fp({"width": 4, "mode": "off", "flag": False,
                                                                    "frac": 0.0, "cats": ()})


@pytest.mark.parametrize("a,b", [
    ({"mode": "on"}, {"mode": "off"}),          # both recorded, non-default difference
    ({"mode": "on"}, {}),                       # absent vs recorded at a NON-default
    ({"width": 3}, {"width": 4}),
])
def test_any_non_default_difference_splits(a, b):
    assert fp(a) != fp(b)


@pytest.mark.parametrize("value", ["none", "OFF", None, False, 0])
def test_another_off_spelling_or_type_is_not_merged_with_the_default(value):
    """``'off'`` is the default; ``'none'`` / ``None`` / ``False`` / ``0`` are other values the constructor may treat
    differently, so a recorded one splits from an absent key."""
    assert fp({"mode": value}) != fp({})


@pytest.mark.parametrize("key,value", [("width", False), ("flag", 0), ("frac", 0), ("cats", [])])
def test_an_equal_value_of_another_type_is_not_a_default(key, value):
    """``0 == False`` and ``0 == 0.0`` in Python; the canonical form needs the same TYPE (and repr) too."""
    assert fp({key: value}) != fp({})


def test_a_key_the_signature_does_not_declare_is_kept():
    assert fp({"unknown": "off"}) != fp({})


def test_an_unreadable_extractor_class_canonicalises_nothing():
    assert canonical_extractor_kwargs(None, {"mode": "off"}) == {"mode": "off"}
    assert fp({"mode": "off"}, cls=None) != fp({}, cls=None)


def test_a_new_kwarg_added_later_with_a_default_does_not_split_old_checkpoints():
    """An OLD checkpoint (no ``speed``) and a NEW one recording ``speed`` at its default, both loaded by a class that
    has since gained it: one fingerprint; and the old checkpoint's own hash is the one it had before (dropping, not
    filling: its canonical kwargs are unchanged)."""
    class _ExtV2:
        def __init__(self, observation_space: Any, mode: str = "off", speed: str = "off"):
            raise AssertionError

    old = {"mode": "on"}
    assert canonical_extractor_kwargs(_ExtV2, old) == old
    assert fp(old, _ExtV2) == fp({"mode": "on", "speed": "off"}, _ExtV2)
    assert fp(old, _ExtV2) != fp({"mode": "on", "speed": "on"}, _ExtV2)


# --- the real extractor's signature ------------------------------------------------------------------------------

REAL = Gen3FeaturesExtractor


def test_the_look3_pair_shares_a_fingerprint_on_the_real_extractor():
    """FAILS ON REVERT: the 706fa536-trained checkpoints record ``policy_readout: 'tower'`` and ``oracle_reveal:
    'off'``; the older ones lack both (``x5ab_look3_2026-10-07`` FINDING 1)."""
    old = {"belief_tokens": "blob", "damage_topk_k": 4}
    new = dict(old, policy_readout="tower", oracle_reveal="off")
    assert fp(old, REAL) == fp(new, REAL)


@pytest.mark.parametrize("key,value", [("policy_readout", "trunk"), ("oracle_reveal", "species"),
                                       ("belief_tokens", "fixed_mass")])
def test_a_real_non_default_value_splits_from_an_absent_key(key, value):
    assert fp({key: value}, REAL) != fp({}, REAL)


def test_a_registry_default_that_is_not_the_signature_default_still_splits():
    """``attend_unrevealed_opponents`` is True in the flag registry (the CLI default) but False in the extractor's
    signature, which is what an ABSENT key builds at load. A checkpoint recording True must not merge with one that
    lacks the key; one recording False must."""
    from agents.model.flag_registry import REGISTRY

    reg = {f.name: f.default for f in REGISTRY}
    assert reg["attend_unrevealed_opponents"] is True                      # the precondition of this test
    assert fp({"attend_unrevealed_opponents": True}, REAL) != fp({}, REAL)
    assert fp({"attend_unrevealed_opponents": False}, REAL) == fp({}, REAL)
