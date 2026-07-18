"""Unit tests for Gen3Teambuilder team-side PFSP (variance-weighted team sampling).

Pure unit tests: the Node team-validator is mocked (so no bridge subprocess), but the team strings
are REAL gen3ou exports, so parse/pack/HP-IV-fix run for real and produce real packed teams.
"""
import random
from unittest import mock

import pytest

from utils.teambuilder import Gen3Teambuilder
from agents.training.team_pfsp_callback import compute_team_pfsp_weights

# Two real gen3ou sample teams (data/teams/sample/). Different packs → distinguishable draws.
TEAM_A = """Jynx (F) @ Leftovers
Ability: Oblivious
EVs: 36 HP / 252 SpA / 220 Spe
Timid Nature
IVs: 0 Atk
- Ice Beam
- Calm Mind
- Substitute
- Lovely Kiss

Suicune @ Leftovers
Ability: Pressure
EVs: 56 HP / 220 SpA / 232 Spe
Timid Nature
IVs: 2 Atk / 30 SpA
- Calm Mind
- Hydro Pump
- Ice Beam
- Hidden Power [Grass]

Dugtrio @ Choice Band
Ability: Arena Trap
EVs: 40 HP / 144 Atk / 100 SpD / 224 Spe
Jolly Nature
- Earthquake
- Beat Up
- Hidden Power [Bug]
- Aerial Ace

Claydol @ Leftovers
Ability: Levitate
EVs: 244 HP / 204 Atk / 32 SpA / 20 SpD / 8 Spe
Adamant Nature
- Rapid Spin
- Earthquake
- Psychic
- Explosion

Gengar @ Leftovers
Ability: Levitate
EVs: 168 HP / 164 SpD / 176 Spe
Timid Nature
- Explosion
- Hidden Power [Grass]
- Thunderbolt
- Will-O-Wisp

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Calm Mind
- Ice Punch
- Thunderbolt
- Substitute
"""

TEAM_B = """Suicune @ Leftovers
Ability: Pressure
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
- Calm Mind
- Surf
- Roar
- Rest

Dugtrio (M) @ Choice Band
Ability: Arena Trap
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
IVs: 30 SpD / 30 Spe
- Earthquake
- Hidden Power [Bug]
- Aerial Ace
- Screech

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 0 Atk
- Thunder Wave
- Seismic Toss
- Aromatherapy
- Soft-Boiled

Claydol @ Leftovers
Ability: Levitate
EVs: 248 HP / 188 Atk / 72 Def
Sassy Nature
- Sunny Day
- Psychic
- Rapid Spin
- Explosion

Snorlax (M) @ Leftovers
Ability: Thick Fat
EVs: 204 HP / 72 Atk / 132 Def / 100 SpD
Careful Nature
- Curse
- Return
- Shadow Ball
- Rest

Forretress (M) @ Leftovers
Ability: Sturdy
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Spikes
- Hidden Power [Ghost]
- Rapid Spin
- Rest
"""


def _make_builder(teams, **kwargs):
    """Construct a Gen3Teambuilder with the Node validator mocked to all-valid."""
    with mock.patch("utils.bridge.team_validator.validate_teams_locally",
                    side_effect=lambda fmt, ts: [{"valid": True} for _ in ts]):
        return Gen3Teambuilder(teams, **kwargs)


def test_team_pfsp_off_is_uniform_rng_identical():
    """OFF mode's yield_team must draw byte-identically to a plain random.choice — i.e. it adds no
    new RNG draws and consumes the stream the exact same way (the byte-identical-default guarantee)."""
    tb = _make_builder([TEAM_A, TEAM_B])  # default team_pfsp="off"
    assert tb._team_pfsp == "off"
    n = 200

    random.seed(0)
    reference = [random.choice(tb.packed_teams) for _ in range(n)]

    random.seed(0)
    drawn = [tb.yield_team() for _ in range(n)]

    assert drawn == reference
    # OFF never tracks a pool index.
    assert tb._last_pool_idx is None


def test_team_pfsp_weighted_sampling():
    """VAR mode honors the pushed weights: a 10.0 vs 0.0001 split draws team 0 overwhelmingly."""
    tb = _make_builder([TEAM_A, TEAM_B], team_pfsp="var")
    tb.set_team_pfsp_weights([10.0, 0.0001])

    random.seed(1)
    idxs = []
    for _ in range(500):
        tb.yield_team()
        idxs.append(tb._last_pool_idx)

    n0 = idxs.count(0)
    n1 = idxs.count(1)
    assert n0 + n1 == 500
    assert n0 > 480, f"team 0 should dominate with a 10 vs 0.0001 weight (got {n0} vs {n1})"


def test_pool_keys_match_team_sha_and_parallel_packed():
    """get_team_pfsp_keys returns sha1(team_str.strip())[:10] (the team_sha convention) per pool team,
    parallel to packed_teams — the identity a worker exposes for the cross-worker GIGO guard + audit."""
    import hashlib
    tb = _make_builder([TEAM_A, TEAM_B], team_pfsp="var")
    keys = tb.get_team_pfsp_keys()
    assert len(keys) == len(tb.packed_teams) == 2
    assert keys[0] == hashlib.sha1(TEAM_A.strip().encode()).hexdigest()[:10]
    assert keys[1] == hashlib.sha1(TEAM_B.strip().encode()).hexdigest()[:10]
    assert keys[0] != keys[1]


def test_record_and_drain():
    """record_team_pfsp_outcome credits the LAST yielded index; drain returns exact counts + zeros."""
    tb = _make_builder([TEAM_A, TEAM_B], team_pfsp="var")
    # Force every draw to index 0 so the recorded outcomes are deterministic.
    tb.set_team_pfsp_weights([1.0, 0.0])

    tb.yield_team()
    assert tb._last_pool_idx == 0
    tb.record_team_pfsp_outcome(1.0)   # a win on idx 0
    tb.yield_team()
    assert tb._last_pool_idx == 0
    tb.record_team_pfsp_outcome(0.0)   # a loss on idx 0

    wins, games, n_pool = tb.drain_team_pfsp_counts()
    assert n_pool == 2
    assert wins == [1.0, 0.0]
    assert games == [2.0, 0.0]

    # A second drain is empty (each pull is one window → zeroed).
    wins2, games2, n2 = tb.drain_team_pfsp_counts()
    assert wins2 == [0.0, 0.0]
    assert games2 == [0.0, 0.0]
    assert n2 == 2


def test_record_is_noop_when_off_or_bias():
    """OFF never records; and a bias-team yield sets _last_pool_idx=None so it isn't tracked."""
    # OFF: record is a hard no-op.
    off = _make_builder([TEAM_A, TEAM_B])  # off
    off.yield_team()
    off.record_team_pfsp_outcome(1.0)
    # No PFSP accumulators moved (they exist but stay zero).
    assert off._tp_games == [0.0, 0.0]

    # VAR + a bias pool: force the bias branch (bias_prob=1.0) → not tracked.
    var = _make_builder([TEAM_A, TEAM_B], bias_teams=[TEAM_B], bias_prob=1.0, team_pfsp="var")
    var.yield_team()
    assert var._last_pool_idx is None
    var.record_team_pfsp_outcome(1.0)   # no-op (idx None)
    assert var._tp_games == [0.0, 0.0]


def test_set_weights_guard_rejects_wrong_length():
    """A None or wrong-length weight vector is ignored (never partially applied)."""
    tb = _make_builder([TEAM_A, TEAM_B], team_pfsp="var")
    tb.set_team_pfsp_weights([3.0, 4.0])
    assert tb._tp_weights == [3.0, 4.0]
    tb.set_team_pfsp_weights(None)              # ignored
    assert tb._tp_weights == [3.0, 4.0]
    tb.set_team_pfsp_weights([1.0, 2.0, 3.0])   # wrong length → ignored
    assert tb._tp_weights == [3.0, 4.0]
    tb.set_team_pfsp_weights([-5.0, 2.0])       # clamps negatives to 0
    assert tb._tp_weights == [0.0, 2.0]


def test_cap_and_floor_weights():
    """The callback's pure weight math: raw = floor + p(1-p), capped at cap*mean(raw)."""
    # Case A (no cap bites): emas [0.5, 0.5, 0.0], floor 0.05, cap 3.0.
    w = compute_team_pfsp_weights([0.5, 0.5, 0.0], floor=0.05, cap=3.0)
    # raw = [0.30, 0.30, 0.05]; mean = 0.216667; cap_val = 0.65 → none capped.
    assert w == pytest.approx([0.30, 0.30, 0.05])

    # Case B (cap bites): one high-variance team among many extremes, tight cap=1.0.
    w2 = compute_team_pfsp_weights([0.5, 0.0, 0.0, 0.0, 0.0], floor=0.05, cap=1.0)
    # raw = [0.30, 0.05, 0.05, 0.05, 0.05]; mean = 0.10; cap_val = 0.10.
    # idx0 raw 0.30 is capped down to 0.10; the 0.05s are unchanged.
    assert w2 == pytest.approx([0.10, 0.05, 0.05, 0.05, 0.05])
    assert w2[0] < 0.30  # the cap bit


def test_cap_and_floor_empty():
    """Degenerate empty EMA list → empty weights (no divide-by-zero)."""
    assert compute_team_pfsp_weights([], floor=0.05, cap=3.0) == []


def test_onesided_weights():
    """'onesided' mode: the LOSING side is held at the MAX weight — w(p)=0.25 for p<0.5, else
    p(1-p) — so a badly-losing team samples like a 50/50 one and only MASTERY retires a team.
    Continuous at 0.5; the unmeasured 0.5 seed gives the max either way."""
    # losing (0.0, 0.2) → held at 0.25; winning (0.8, 1.0) → variance decay (0.16, 0.0).
    w = compute_team_pfsp_weights([0.0, 0.2, 0.5, 0.8, 1.0], floor=0.05, cap=100.0, onesided=True)
    assert w == pytest.approx([0.30, 0.30, 0.30, 0.21, 0.05])
    # default (symmetric) still decays the losing side — the two modes differ only there.
    ws = compute_team_pfsp_weights([0.0, 0.2, 0.5, 0.8, 1.0], floor=0.05, cap=100.0)
    assert ws == pytest.approx([0.05, 0.21, 0.30, 0.21, 0.05])
    # continuity at the boundary: p just below/at 0.5 give ~equal weight.
    lo, hi = compute_team_pfsp_weights([0.4999, 0.5], floor=0.0, cap=100.0, onesided=True)
    assert lo == pytest.approx(hi, abs=1e-3)
