"""The ``--team-mix`` draw inside ``Gen3Teambuilder`` (``gen3_team_mix_v1``; designs/training/team_curriculum.md
"The team mix").

Pure unit tests: the Node validator is mocked (``teambuilder_test._make_builder``), the team strings are REAL gen3ou
exports, so parse / pack / the Hidden-Power fix run for real. Each test names the revert it fails on.
"""
import pickle
import random
from unittest import mock

import pytest

from utils.teambuilder import Gen3Teambuilder
from utils.teambuilder_test import TEAM_A, TEAM_B, _make_builder

#: The instance attributes the mix may add — none of them may exist on a builder WITHOUT a mix.
_MIX_ATTRS = ("mix_packed_teams", "mix_prob", "mix_dropped")


def test_no_mix_builder_carries_no_mix_state_and_pickles_unchanged():
    """A builder without a mix has no mix attribute in ``__dict__`` (the class defaults answer), so its pickle — what
    every env worker unpickles — is the pre-mix one. Fails on revert to setting the attributes unconditionally in
    ``__init__``."""
    tb = _make_builder([TEAM_A, TEAM_B], bias_teams=[TEAM_B], bias_prob=0.1)
    assert not set(_MIX_ATTRS) & set(vars(tb))
    assert tb.mix_packed_teams == () and tb.mix_prob == 0.0 and tb.mix_dropped == ()
    clone = pickle.loads(pickle.dumps(tb))
    assert not set(_MIX_ATTRS) & set(vars(clone))
    # an explicit mix_prob without teams, or teams at prob 0, is NO mix either
    for kw in ({"mix_teams": None, "mix_prob": 0.5}, {"mix_teams": [TEAM_B], "mix_prob": 0.0}):
        assert not set(_MIX_ATTRS) & set(vars(_make_builder([TEAM_A], **kw)))


def test_no_mix_draw_stream_is_the_legacy_bias_then_pool_stream():
    """Without a mix the draw consumes EXACTLY the legacy calls: ``random()`` for the bias branch, then one
    ``choice``. The reference is written out by hand from the global stream. Fails on revert to a mix branch that
    draws its ``random()`` even when no mix is declared (an extra RNG call shifts every later draw)."""
    tb = _make_builder([TEAM_A, TEAM_B], bias_teams=[TEAM_B], bias_prob=0.3)
    random.seed(5)
    ref = []
    for _ in range(300):
        ref.append(random.choice(tb.bias_packed_teams) if random.random() < 0.3 else random.choice(tb.packed_teams))
    random.seed(5)
    assert [tb.yield_team() for _ in range(300)] == ref


def _mixed(pool, mix, prob, **kw):
    return _make_builder(pool, mix_teams=mix, mix_prob=prob, **kw)


def test_the_mix_branch_comes_first_then_the_unchanged_draw():
    """A seeded builder's stream: ``random() < share`` → a corpus team, else the bias / pool draw exactly as
    before. Fails on revert to placing the mix AFTER the bias branch, or to any other order of RNG calls."""
    tb = _mixed([TEAM_A], [TEAM_B], 0.5, bias_teams=[TEAM_A], bias_prob=0.1, rng_seed=17)
    ref_rng = random.Random(17)
    ref = []
    for _ in range(400):
        if ref_rng.random() < 0.5:
            ref.append(("mix", ref_rng.choice(tb.mix_packed_teams)))
        elif ref_rng.random() < 0.1:
            ref.append(("bias", ref_rng.choice(tb.bias_packed_teams)))
        else:
            ref.append(("pool", ref_rng.choice(tb.packed_teams)))
    got = []
    for kind, team in ref:
        t = tb.yield_team()
        got.append((kind, t))
        # a corpus or bias draw is untracked; a pool draw names its pool index
        assert (tb._last_pool_idx is None) == (kind != "pool")
    assert got == ref
    kinds = [k for k, _ in ref]
    assert {"mix", "bias", "pool"} <= set(kinds)   # the precondition: every branch was exercised


def test_a_corpus_draw_is_untracked_by_the_per_team_wr_tracker():
    """Every yield at share 1.0 is a corpus team with ``_last_pool_idx`` None, so the WR tracker records nothing.
    Fails on revert to leaving ``_last_pool_idx`` at its previous value on a corpus draw."""
    tb = _mixed([TEAM_A], [TEAM_B], 1.0, rng_seed=3)
    tb._last_pool_idx = 0                                   # a stale pool index from an earlier draw
    for _ in range(20):
        assert tb.yield_team() in tb.mix_packed_teams
        assert tb._last_pool_idx is None
        tb.record_team_wr_outcome(1.0, 0, 4)
    counts, _ = tb.drain_team_wr_counts()
    assert counts == {}


def test_block_episodes_hold_a_corpus_draw():
    """Team blocking holds the WHOLE draw, a corpus draw included, untracked for the whole block."""
    tb = _mixed([TEAM_A], [TEAM_B], 1.0, rng_seed=9)
    tb.set_block_episodes(4)
    seq = [tb.yield_team() for _ in range(8)]
    assert len(set(seq[:4])) == 1 and len(set(seq[4:])) == 1 and seq[0] in tb.mix_packed_teams
    assert tb._last_pool_idx is None


def test_mix_validation_drops_are_recorded_and_packing_is_the_pools():
    """Corpus teams go through the SAME validator + pack as the pool; a rejected one is listed in ``mix_dropped``
    (the trainer's startup guard refuses any). Fails on revert to skipping validation for the corpus."""
    def verdicts(fmt, ts):
        return [{"valid": "Jynx" not in t, "errors": ["Jynx is banned here"]} for t in ts]

    with mock.patch("utils.bridge.team_validator.validate_teams_locally", side_effect=verdicts):
        tb = Gen3Teambuilder([TEAM_B], mix_teams=[TEAM_A, TEAM_B], mix_prob=0.5)
    assert tb.mix_dropped == [(0, ["Jynx is banned here"])]
    assert tb.mix_packed_teams == tuple(tb.packed_teams)    # TEAM_B packed exactly as the pool packs it
    assert tb.mix_pool_copies() == 1


def test_mix_refuses_a_bad_share_and_an_empty_survivor_set():
    with pytest.raises(ValueError, match=r"mix_prob must be in \(0, 1\]"):
        _mixed([TEAM_A], [TEAM_B], 1.5)
    # first call: the pool (valid); second: the corpus (every team rejected)
    with mock.patch("utils.bridge.team_validator.validate_teams_locally",
                    side_effect=[[{"valid": True}], [{"valid": False, "errors": ["no"]}]]):
        with pytest.raises(ValueError, match="no corpus team passes"):
            Gen3Teambuilder([TEAM_A], mix_teams=[TEAM_B], mix_prob=0.5)


def test_mix_share_lands_deterministically():
    """At share 0.5 over 2,000 seeded draws the corpus count is the EXACT count of the seeded reference stream (a
    deterministic check, rule 8 — no tolerance by chance) and that fixed number lies in [900, 1100]."""
    tb = _mixed([TEAM_A], [TEAM_B], 0.5, rng_seed=2026)
    n_mix = 0
    for _ in range(2000):
        tb.yield_team()
        n_mix += tb._last_pool_idx is None
    ref_rng = random.Random(2026)
    ref = 0
    for _ in range(2000):
        hit = ref_rng.random() < 0.5
        ref += hit
        ref_rng.choice(tb.mix_packed_teams if hit else tb.packed_teams)
    assert n_mix == ref
    assert 900 <= n_mix <= 1100, n_mix
