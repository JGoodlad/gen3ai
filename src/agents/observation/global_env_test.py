"""GlobalEnvEncoder tests — LiveView API (event-sourced weather + hazards + screens)."""

import math

import numpy as np
import pytest

from agents.observation.global_env import GlobalEnvEncoder, _WEATHER_MAX_TURNS
from agents.observation.constants import GLOBAL_ENV_DIM, WEATHER_ONEHOT_DIM, MAX_TURNS, MAX_SPIKES
from agents.battle.live_view import LiveView, LiveSide, LiveWeather


def _view(turn=1, weather=None, ours_sc=None, opp_sc=None):
    def side(sc):
        return LiveSide(team_size=6, active=None, mons=(), side_conditions=sc or {})
    return LiveView(
        turn=turn,
        weather=weather or LiveWeather(None, False, 0),
        ours=side(ours_sc),
        opp=side(opp_sc),
    )


def test_dimension_matches_layout():
    g = GlobalEnvEncoder()
    L = g.get_layout()
    assert g.dimension == GLOBAL_ENV_DIM
    assert sum(p["dim"] for p in L.values()) == GLOBAL_ENV_DIM


def test_global_env_weather():
    vec = GlobalEnvEncoder().encode(_view(
        weather=LiveWeather("sunnyday", is_permanent=False, turns_active=0)))
    assert vec[1] == 1.0  # SUN one-hot index 1
    assert vec[0] == 0.0  # NONE


def test_weather_none_is_finite():
    vec = GlobalEnvEncoder().encode(_view())
    assert vec[0] == 1.0
    assert np.all(np.isfinite(vec))


def test_weather_move_sourced_countdown():
    vec = GlobalEnvEncoder().encode(_view(
        weather=LiveWeather("raindance", is_permanent=False, turns_active=1)))
    assert vec[2] == 1.0                       # RAIN one-hot
    assert vec[WEATHER_ONEHOT_DIM] == 0.0      # not permanent
    assert vec[WEATHER_ONEHOT_DIM + 1] == pytest.approx((5 - 1) / _WEATHER_MAX_TURNS)


def test_weather_ability_sourced_permanent():
    vec = GlobalEnvEncoder().encode(_view(
        weather=LiveWeather("sandstorm", is_permanent=True, turns_active=9)))
    assert vec[3] == 1.0                       # SAND one-hot
    assert vec[WEATHER_ONEHOT_DIM] == 1.0      # permanent
    assert vec[WEATHER_ONEHOT_DIM + 1] == 0.0  # no finite countdown


def test_global_env_hazards():
    vec = GlobalEnvEncoder().encode(_view(ours_sc={"spikes": 2}, opp_sc={"spikes": 3}))
    hz = WEATHER_ONEHOT_DIM + 2
    assert vec[hz] == pytest.approx(2 / MAX_SPIKES)
    assert vec[hz + 1] == 1.0  # 3/3


def test_global_env_clock():
    vec = GlobalEnvEncoder().encode(_view(turn=10))
    ck = WEATHER_ONEHOT_DIM + 4
    assert vec[ck] == pytest.approx(math.log(1 + 10) / math.log(1 + MAX_TURNS))


# --------------------------------------------------------------------------- #
# gen3_deadline_clock_v1 — the CLOCK group is [log_elapsed, remaining_linear, log_remaining]
# --------------------------------------------------------------------------- #
def test_clock_group_carries_turns_remaining():
    """The two REMAINING channels encode `MAX_TURNS - turn` (the forfeit deadline), linearly and
    log-scaled, and both round-trip through describe_vector."""
    ck = WEATHER_ONEHOT_DIM + 4
    enc = GlobalEnvEncoder()
    for turn in (1, 50, 200, 240, 249):
        rem = MAX_TURNS - turn
        vec = enc.encode(_view(turn=turn))
        assert vec[ck] == pytest.approx(math.log(1 + turn) / math.log(1 + MAX_TURNS))
        assert vec[ck + 1] == pytest.approx(rem / MAX_TURNS)
        assert vec[ck + 2] == pytest.approx(math.log(1 + rem) / math.log(1 + MAX_TURNS))
        d = enc.describe_vector(vec)
        assert d["turns_remaining"] == pytest.approx(rem, abs=0.1)
        assert d["turns_remaining_log"] == pytest.approx(rem, abs=0.1)


def test_clock_clamps_at_and_past_the_deadline():
    """At the cap both remaining channels are exactly 0, and an OVER-cap turn saturates there
    rather than going negative (a linear channel) or NaN (log of a non-positive remaining)."""
    ck = WEATHER_ONEHOT_DIM + 4
    for turn in (MAX_TURNS, MAX_TURNS + 5, MAX_TURNS + 100):
        vec = GlobalEnvEncoder().encode(_view(turn=turn))
        assert vec[ck + 1] == 0.0
        assert vec[ck + 2] == 0.0
        assert np.isfinite(vec).all()


def test_log_remaining_has_resolution_AT_the_deadline():
    """The POINT of the channel: the last 20 turns must occupy a large share of log-remaining's
    range, where they occupy ~1.5% of log-elapsed's. Without this the critic has no resolution on
    the cliff TD has to fit first — the measured ai_v9_09 failure (13/14 timeouts had a POSITIVE
    V on the last decision before a -30 forfeit)."""
    ck = WEATHER_ONEHOT_DIM + 4
    enc = GlobalEnvEncoder()
    at230, at250 = enc.encode(_view(turn=230)), enc.encode(_view(turn=MAX_TURNS))
    elapsed_share = abs(at250[ck] - at230[ck]) / at250[ck]
    log_rem_share = abs(at230[ck + 2] - at250[ck + 2]) / enc.encode(_view(turn=0))[ck + 2]
    assert elapsed_share < 0.03                      # the old feature: ~1.5% over the last 20 turns
    assert log_rem_share > 0.50                      # the new one: ~55%
    assert log_rem_share > 20 * elapsed_share        # >=20x more resolution where it matters


def test_max_turns_is_the_forfeit_deadline():
    """The obs clock normaliser and the turn the trainee actually FORFEITS on are ONE number.
    They were independently-written 250s in two files; moving the stall threshold would have
    silently mis-scaled the `turns_remaining` scalars the critic prices the deadline with."""
    from agents.training.stall import StallConfig
    assert StallConfig().threshold == MAX_TURNS


def test_global_env_screens():
    # LiveView.side_conditions key on SideCondition.name.lower() → "light_screen"
    vec = GlobalEnvEncoder().encode(_view(
        ours_sc={"reflect": 1, "safeguard": 1},
        opp_sc={"light_screen": 1, "mist": 1}))
    d = GlobalEnvEncoder().describe_vector(vec)
    assert d["our_reflect"] is True and d["our_safeguard"] is True
    assert d["opp_light_screen"] is True and d["opp_mist"] is True
    assert d["our_light_screen"] is False and d["opp_reflect"] is False


def test_global_env_screens_none():
    d = GlobalEnvEncoder().describe_vector(GlobalEnvEncoder().encode(_view()))
    for k in ("our_reflect", "our_light_screen", "opp_reflect", "opp_light_screen",
              "our_safeguard", "opp_mist"):
        assert d[k] is False


def test_describe_vector_weather():
    vec = GlobalEnvEncoder().encode(_view(
        weather=LiveWeather("sandstorm", is_permanent=True, turns_active=3), turn=5))
    d = GlobalEnvEncoder().describe_vector(vec)
    assert d["weather"] == "SAND"
    assert d["weather_permanent"] is True
    assert round(d["turn"]) == 5
