from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.weather import Weather
from unittest.mock import MagicMock

battle = MagicMock(spec=AbstractBattle)
battle.weather = {Weather.SUNNYDAY: 5} # Maybe?
print(type(battle.weather))
