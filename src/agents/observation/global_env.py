import numpy as np
import math
from .base import ObservationEncoder
from .constants import GLOBAL_ENV_DIM, MAX_TURNS, MAX_SPIKES
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from typing import Any, Dict

class GlobalEnvEncoder(ObservationEncoder):
    """
    Encodes global environment data (weather, hazards, clock).
    """

    @property
    def dimension(self) -> int:
        return GLOBAL_ENV_DIM

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)

        # 1. Weather (6 dims: 0=none, 1=sun, 2=rain, 3=sand, 4=hail, 5=unused)
        weather_map = {
            Weather.SUNNYDAY: 1, Weather.DESOLATELAND: 1,
            Weather.RAINDANCE: 2, Weather.PRIMORDIALSEA: 2,
            Weather.SANDSTORM: 3, Weather.HAIL: 4, Weather.SNOW: 4
        }

        current_weather = None
        if battle.weather:
            current_weather = next(iter(battle.weather.keys()))

        idx = weather_map.get(current_weather, 0)
        vec[idx] = 1.0

        # 2. Hazards (2)
        p1_spikes = battle.side_conditions.get(SideCondition.SPIKES, 0)
        p2_spikes = battle.opponent_side_conditions.get(SideCondition.SPIKES, 0)
        vec[6] = float(p1_spikes) / MAX_SPIKES
        vec[7] = float(p2_spikes) / MAX_SPIKES

        # 3. Clock (1)
        turn = battle.turn
        vec[8] = math.log(1 + turn) / math.log(1 + MAX_TURNS)

        # 4. Screens (4): our reflect, our light screen, opp reflect, opp light screen
        vec[9]  = 1.0 if SideCondition.REFLECT in battle.side_conditions else 0.0
        vec[10] = 1.0 if SideCondition.LIGHT_SCREEN in battle.side_conditions else 0.0
        vec[11] = 1.0 if SideCondition.REFLECT in battle.opponent_side_conditions else 0.0
        vec[12] = 1.0 if SideCondition.LIGHT_SCREEN in battle.opponent_side_conditions else 0.0

        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "weather": {"offset": 0, "dim": 6},
            "hazards": {"offset": 6, "dim": 2},
            "clock": {"offset": 8, "dim": 1},
            "screens": {"offset": 9, "dim": 4}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Weather
        weather_names = ["NONE", "SUN", "RAIN", "SAND", "HAIL"]
        weather = "NONE"
        for i in range(5):
            if vector[i] > 0.5:
                weather = weather_names[i]
                break

        # Hazards
        our_spikes = int(vector[6] * MAX_SPIKES)
        opp_spikes = int(vector[7] * MAX_SPIKES)

        # Turn
        turn_norm = vector[8]
        turn = math.exp(turn_norm * math.log(1 + MAX_TURNS)) - 1

        return {
            "weather": weather,
            "our_spikes": our_spikes,
            "opp_spikes": opp_spikes,
            "turn": round(turn, 1),
            "our_reflect": bool(vector[9] > 0.5),
            "our_light_screen": bool(vector[10] > 0.5),
            "opp_reflect": bool(vector[11] > 0.5),
            "opp_light_screen": bool(vector[12] > 0.5)
        }
