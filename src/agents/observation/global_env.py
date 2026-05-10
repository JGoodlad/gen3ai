import numpy as np
import math
from .base import ObservationEncoder
from .constants import GLOBAL_ENV_DIM, MAX_TURNS, MAX_SPIKES
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.weather import Weather
from typing import Any

class GlobalEnvEncoder(ObservationEncoder):
    """
    Encodes global environment data (weather, hazards, clock).
    """

    @property
    def dimension(self) -> int:
        return GLOBAL_ENV_DIM

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        
        cursor = 0
        
        # 1. Weather (5 + 1 = 6)
        # None, Sun, Rain, Sand, Hail
        weather_map = {
            Weather.SUNNYDAY: 1, Weather.DESOLATELAND: 1,
            Weather.RAINDANCE: 2, Weather.PRIMORDIALSEA: 2,
            Weather.SANDSTORM: 3, Weather.HAIL: 4, Weather.SNOW: 4
        }
        
        # In poke-env, battle.weather is a Dict[Weather, int] (turns remaining)
        current_weather = None
        if battle.weather:
            current_weather = next(iter(battle.weather.keys()))
            
        idx = weather_map.get(current_weather, 0)
        if idx >= 0:
            vec[cursor + idx] = 1.0
        
        # Weather turns remaining (Normalized)
        # In poke-env, weather_duration is not always easy to get without tracking.
        # Placeholder.
        cursor += 6
        
        # 2. Hazards (2)
        # P1 Spikes (1), P2 Spikes (1)
        p1_spikes = battle.side_conditions.get("spikes", 0)
        p2_spikes = battle.opponent_side_conditions.get("spikes", 0)
        vec[cursor] = float(p1_spikes) / MAX_SPIKES
        vec[cursor+1] = float(p2_spikes) / MAX_SPIKES
        cursor += 2
        
        # 3. Clock (3)
        # Turn count (ln(1+T) / ln(1001))
        turn = battle.turn
        vec[cursor] = math.log(1 + turn) / math.log(1 + MAX_TURNS)
        
        # Screen R (Reflect), LS (Light Screen)
        # linear normalized T_rem / 5
        # side_conditions value is usually 1 (active) but poke-env might store turns.
        # In ADV, screens last 5 turns.
        cursor += 1
        # TODO: Implement screen turn tracking.
        
        return vec
