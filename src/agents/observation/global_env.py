import numpy as np
import math
from .base import ObservationEncoder
from .constants import GLOBAL_ENV_DIM, WEATHER_ONEHOT_DIM, MAX_TURNS, MAX_SPIKES, CLOCK_DIM
from typing import Any, Dict


# Weather id (LiveWeather.weather, poke-env id-form) -> one-hot index. Only gen3ou
# weathers; index 0 = none. Gen4+ weather slots are not encoded (no wasted dims).
_WEATHER_IDX = {
    None: 0,
    "sunnyday": 1, "desolateland": 1,
    "raindance": 2, "primordialsea": 2,
    "sandstorm": 3,
    "hail": 4, "snow": 4, "snowscape": 4,
}
_WEATHER_NAMES = ["NONE", "SUN", "RAIN", "SAND", "HAIL"]
# gen3 move-set weather lasts 5 turns; used to normalise turns-remaining.
_WEATHER_MAX_TURNS = 5

# Shared denominator for BOTH clock log scalars, so elapsed and remaining are on ONE scale
# (log(1+MAX_TURNS)); precomputed because encode() is the hot per-decision path.
_LOG_MAX_TURNS = math.log(1 + MAX_TURNS)

# Per-side side-conditions encoded as presence bits (gen3ou), as the lower-cased ``SideCondition`` names the
# LiveView keys its ``side_conditions`` by (``reflect``, ``light_screen``, ``safeguard``, ``mist``). Spelled as ids
# so this module imports no poke-env (P1 of the retirement); ``global_env_test.py`` pins them to the fork's enum.
# Spikes is a 0-3 count handled separately.
_SCREEN_CONDITIONS = ["reflect", "light_screen", "safeguard", "mist"]


class GlobalEnvEncoder(ObservationEncoder):
    """Encodes field state from a :class:`~agents.battle.live_view.LiveView`: weather
    (with permanence + turns-remaining, both event-sourced — never guessed), hazards,
    the turn clock, and per-side screens / Safeguard / Mist.
    """

    @property
    def dimension(self) -> int:
        return GLOBAL_ENV_DIM

    # Why the `type: ignore[override]` below — LiveView-subject encoder; see ActiveContextEncoder.encode.
    def get_layout(self) -> Dict[str, Any]:
        # Offsets must match encode(); the feature extractor slices weather/hazards/
        # clock from here. "weather" spans the one-hot + permanence + turns (7 dims) so
        # the network's broadcast weather feature carries the full weather state.
        return {
            "weather": {"offset": 0, "dim": WEATHER_ONEHOT_DIM + 2},
            "hazards": {"offset": WEATHER_ONEHOT_DIM + 2, "dim": 2},
            "clock": {"offset": WEATHER_ONEHOT_DIM + 4, "dim": CLOCK_DIM},
            "screens": {"offset": WEATHER_ONEHOT_DIM + 4 + CLOCK_DIM, "dim": 8},
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        weather = "NONE"
        for i in range(WEATHER_ONEHOT_DIM):
            if vector[i] > 0.5:
                weather = _WEATHER_NAMES[i]
                break
        permanent = bool(vector[WEATHER_ONEHOT_DIM] > 0.5)
        turns_left = round(float(vector[WEATHER_ONEHOT_DIM + 1]) * _WEATHER_MAX_TURNS, 1)

        hz = WEATHER_ONEHOT_DIM + 2
        our_spikes = int(round(vector[hz] * MAX_SPIKES))
        opp_spikes = int(round(vector[hz + 1] * MAX_SPIKES))
        turn = math.exp(vector[hz + 2] * _LOG_MAX_TURNS) - 1
        turns_remaining = round(float(vector[hz + 3]) * MAX_TURNS, 1)
        turns_remaining_log = round(math.exp(float(vector[hz + 4]) * _LOG_MAX_TURNS) - 1, 1)

        sc = hz + 2 + CLOCK_DIM
        screens = {}
        for j, cid in enumerate(_SCREEN_CONDITIONS):
            screens[f"our_{cid}"] = bool(vector[sc + j * 2] > 0.5)
            screens[f"opp_{cid}"] = bool(vector[sc + j * 2 + 1] > 0.5)

        return {
            "weather": weather,
            "weather_permanent": permanent,
            "weather_turns_left": turns_left,
            "our_spikes": our_spikes,
            "opp_spikes": opp_spikes,
            "turn": round(turn, 1),
            # gen3_deadline_clock_v1: both decode back to the SAME quantity (turns left before the
            # forfeit) from the linear and the log-remaining channel — a cheap cross-check that the
            # two are consistent, and the prober's FIELD line reads `turns_remaining`.
            "turns_remaining": turns_remaining,
            "turns_remaining_log": turns_remaining_log,
            **screens,
        }
