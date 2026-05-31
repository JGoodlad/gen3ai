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
