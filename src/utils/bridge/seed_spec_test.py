"""Producer-side seed guard — `gen3_bridge_seed_forms_v1`.

The bug class these pin: a bridge ``START`` seed the child could not parse was DROPPED
silently, so the battle ran on some other dice stream while every log said it was seeded.
The fix guards BOTH ends — the rust child now `__ERR__`s (see
``src/rust_sim/tests/sim_bridge_seed_test.rs``) and the Python producers throw here, which
is the half that names the offending CALLER.

Pure unit tests: no bridge child, no battles.
"""

from __future__ import annotations

import pytest

from utils.bridge.seed_spec import validate_seed_spec


@pytest.mark.parametrize("seed,canonical", [
    ([1, 2, 3, 4], "1,2,3,4"),
    ((0, 0, 0, 0), "0,0,0,0"),
    ([65535, 65535, 65535, 65535], "65535,65535,65535,65535"),
    ("1,2,3,4", "1,2,3,4"),
    (" 1,2,3,4 ", "1,2,3,4"),
    ("gen5,0001000200030004", "gen5,0001000200030004"),
    ("sodium,deadbeef", "sodium,deadbeef"),
])
def test_accepts_every_form_new_prng_accepts(seed, canonical):
    assert validate_seed_spec(seed) == canonical


def test_absent_seed_is_legitimate():
    """`None` is the TRAINING/EVAL default and must not throw — the child mints one."""
    assert validate_seed_spec(None) is None


@pytest.mark.parametrize("seed", [
    "",                     # empty string
    "not-a-seed",           # unrecognized prefix
    "1,2,3",                # too few words
    "1,2,3,4,5",            # too many
    "1,2,x,4",              # non-numeric word
    "99999,1,1,1",          # out of the u16 range
    "gen5,zz",              # short/non-hex gen5 payload
    "sodium,",              # empty sodium payload
    "sodium,xyz",           # non-hex sodium payload
    "[1,2,3,4]",            # the bracketed spelling is a JSON artifact, not a seed string
    [1, 2, 3],              # wrong arity
    [1, 2, 3, 4, 5],
    [1, 2, "3", 4],         # a stringly-typed element
    [1, 2, 3, 70000],       # out of range
    12345,                  # a bare number
    True,                   # a bool (an int subclass — must NOT sneak through)
    {"seed": [1, 2, 3, 4]},  # a dict
])
def test_rejects_anything_the_child_could_not_parse(seed):
    with pytest.raises(ValueError):
        validate_seed_spec(seed)


def test_error_names_the_field():
    """`resumeReseed.seed` has a different producer than `seed`; the message must say which."""
    with pytest.raises(ValueError, match="resumeReseed.seed"):
        validate_seed_spec("zzz", what="resumeReseed.seed")
