"""Unit tests for Gen3Teambuilder's pool draw, draw-index attribution and team blocking (team-side
PFSP — variance-weighted team sampling — was deleted with the lever, deletion pass L4).

Pure unit tests: the Node team-validator is mocked (so no bridge subprocess), but the team strings
are REAL gen3ou exports, so parse/pack/HP-IV-fix run for real and produce real packed teams.
"""
import random
from unittest import mock

from utils.teambuilder import Gen3Teambuilder

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


def test_the_pool_draw_is_uniform_rng_identical():
    """The pool draw must be byte-identical to a plain random.choice — i.e. it adds no new RNG draws
    and consumes the stream the exact same way (the byte-identical-default guarantee)."""
    tb = _make_builder([TEAM_A, TEAM_B])
    n = 200

    random.seed(0)
    reference = [random.choice(tb.packed_teams) for _ in range(n)]

    random.seed(0)
    drawn = [tb.yield_team() for _ in range(n)]

    assert drawn == reference
    # The drawn index is RESOLVED by dict lookup (consuming no RNG — that identity is what the
    # assertion above pins) so the always-on per-team WR tracker can attribute the episode.
    assert tb.packed_teams[tb._last_pool_idx] == drawn[-1]


def test_pool_keys_match_team_sha_and_parallel_packed():
    """get_pool_team_keys returns sha1(team_str.strip())[:10] (the team_sha convention) per pool team,
    parallel to packed_teams — the identity a worker exposes for the cross-worker GIGO guard + audit."""
    import hashlib
    tb = _make_builder([TEAM_A, TEAM_B])
    keys = tb.get_pool_team_keys()
    assert len(keys) == len(tb.packed_teams) == 2
    assert keys[0] == hashlib.sha1(TEAM_A.strip().encode()).hexdigest()[:10]
    assert keys[1] == hashlib.sha1(TEAM_B.strip().encode()).hexdigest()[:10]
    assert keys[0] != keys[1]


def test_a_bias_team_yield_is_not_tracked():
    """A bias-team yield sets _last_pool_idx=None, so the per-team WR tracker records nothing for it
    (a pinned bias team is not a pool member)."""
    tb = _make_builder([TEAM_A, TEAM_B], bias_teams=[TEAM_B], bias_prob=1.0)
    tb.yield_team()
    assert tb._last_pool_idx is None
    tb.record_team_wr_outcome(1.0, 0, 4)            # a no-op: idx None
    counts, keys = tb.drain_team_wr_counts()
    assert counts == {} and len(keys) == 2


def test_block_episodes_holds_and_redraws():
    """Team blocking: the same team is yielded K consecutive times, then a fresh draw happens;
    the tracking index stays pinned to the block's team for every yield of the block."""
    tb = _make_builder([TEAM_A, TEAM_B])
    tb.set_block_episodes(3)
    random.seed(7)
    teams = [tb.yield_team() for _ in range(9)]
    # 3 blocks of 3 identical yields each.
    for b in range(3):
        assert teams[3 * b] == teams[3 * b + 1] == teams[3 * b + 2]
    # tracking index constant within a block (outcome attribution) — record 3 outcomes, all land
    # on ONE team.
    tb2 = _make_builder([TEAM_A, TEAM_B])
    tb2.set_block_episodes(3)
    random.seed(11)
    for _ in range(3):
        tb2.yield_team()
        tb2.record_team_wr_outcome(1.0, 0, 4)
    counts, _keys = tb2.drain_team_wr_counts()
    games = [sum(g) for _w, g in counts.values()]
    assert sum(games) == 3.0 and max(games) == 3.0    # all 3 games on the same pool team


def test_block_episodes_off_is_byte_identical():
    """K=1 (default) must take the exact legacy path — identical RNG stream and draws."""
    tb_a = _make_builder([TEAM_A, TEAM_B])
    tb_b = _make_builder([TEAM_A, TEAM_B])
    tb_b.set_block_episodes(1)                        # explicit off
    random.seed(42)
    seq_a = [tb_a.yield_team() for _ in range(6)]
    random.seed(42)
    seq_b = [tb_b.yield_team() for _ in range(6)]
    assert seq_a == seq_b
