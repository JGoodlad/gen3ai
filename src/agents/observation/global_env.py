import numpy as np
import math
from .base import ObservationEncoder
from .constants import GLOBAL_ENV_DIM, WEATHER_ONEHOT_DIM, MAX_TURNS, MAX_SPIKES
from poke_env.battle.side_condition import SideCondition
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

# Per-side side-conditions encoded as presence bits (gen3ou). Spikes is a 0-3 count
# handled separately.
_SCREEN_CONDITIONS = [
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.SAFEGUARD,
    SideCondition.MIST,
]


class GlobalEnvEncoder(ObservationEncoder):
    """Encodes field state from a :class:`~agents.battle.live_view.LiveView`: weather
    (with permanence + turns-remaining, both event-sourced — never guessed), hazards,
    the turn clock, and per-side screens / Safeguard / Mist.
    """

    @property
    def dimension(self) -> int:
        return GLOBAL_ENV_DIM

    def encode(self, live_view: Any) -> np.ndarray:
        """``live_view`` is a :class:`LiveView`."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        w = live_view.weather

        cur = 0
        # 1. Weather one-hot (5)
        vec[cur + _WEATHER_IDX.get(w.weather, 0)] = 1.0
        cur += WEATHER_ONEHOT_DIM

        # 2. Weather permanence + turns-remaining (2). Ability weather is permanent
        #    (turns_remaining None → 0 remaining + permanent=1); move weather counts
        #    down; no weather → both 0. All from the event log — no ability guessing.
        vec[cur] = 1.0 if w.is_permanent else 0.0
        tr = w.turns_remaining
        vec[cur + 1] = (tr / _WEATHER_MAX_TURNS) if tr is not None else 0.0
        cur += 2

        # 3. Hazards: Spikes count per side (2)
        ours, opp = live_view.ours.side_conditions, live_view.opp.side_conditions
        vec[cur] = float(ours.get("spikes", 0)) / MAX_SPIKES
        vec[cur + 1] = float(opp.get("spikes", 0)) / MAX_SPIKES
        cur += 2

        # 4. Turn clock (1, log-normalised)
        vec[cur] = math.log(1 + live_view.turn) / math.log(1 + MAX_TURNS)
        cur += 1

        # 5. Per-side screens / Safeguard / Mist (8): for each condition, [ours, opp]
        for cond in _SCREEN_CONDITIONS:
            cid = cond.name.lower()
            vec[cur] = 1.0 if cid in ours else 0.0
            vec[cur + 1] = 1.0 if cid in opp else 0.0
            cur += 2

        return vec

    def get_layout(self) -> Dict[str, Any]:
        # Offsets must match encode(); the feature extractor slices weather/hazards/
        # clock from here. "weather" spans the one-hot + permanence + turns (7 dims) so
        # the network's broadcast weather feature carries the full weather state.
        return {
            "weather": {"offset": 0, "dim": WEATHER_ONEHOT_DIM + 2},
            "hazards": {"offset": WEATHER_ONEHOT_DIM + 2, "dim": 2},
            "clock": {"offset": WEATHER_ONEHOT_DIM + 4, "dim": 1},
            "screens": {"offset": WEATHER_ONEHOT_DIM + 5, "dim": 8},
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
        turn = math.exp(vector[hz + 2] * math.log(1 + MAX_TURNS)) - 1

        sc = hz + 3
        screens = {}
        for j, cond in enumerate(_SCREEN_CONDITIONS):
            cid = cond.name.lower()
            screens[f"our_{cid}"] = bool(vector[sc + j * 2] > 0.5)
            screens[f"opp_{cid}"] = bool(vector[sc + j * 2 + 1] > 0.5)

        return {
            "weather": weather,
            "weather_permanent": permanent,
            "weather_turns_left": turns_left,
            "our_spikes": our_spikes,
            "opp_spikes": opp_spikes,
            "turn": round(turn, 1),
            **screens,
        }
