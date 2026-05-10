import asyncio
from poke_env.player import RandomPlayer
from poke_env.teambuilder import Teambuilder
from gen3_utils import fix_gen3_hp_ivs

# Team from https://pokepast.es/f6229d2c867e21d6
STAR_TSS_TEAM = """
Skarmory (F) @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 8 Def / 248 SpD
Careful Nature
IVs: 0 Atk
- Spikes
- Protect
- Roar
- Toxic

Blissey @ Leftovers
Ability: Natural Cure
Shiny: Yes
EVs: 252 Def / 252 SpA / 4 Spe
Modest Nature
IVs: 0 Atk
- Soft-Boiled
- Ice Beam
- Toxic
- Fire Blast

Tyranitar (F) @ Leftovers
Ability: Sand Stream
EVs: 248 HP / 196 Atk / 12 Def / 52 SpD
Adamant Nature
- Focus Punch
- Rock Slide
- Hidden Power [Bug]
- Earthquake

Swampert (F) @ Leftovers
Ability: Torrent
EVs: 240 HP / 136 Def / 40 SpA / 48 SpD / 44 Spe
Relaxed Nature
- Earthquake
- Ice Beam
- Hydro Pump
- Protect

Gengar (F) @ Leftovers
Ability: Levitate
EVs: 168 HP / 164 SpD / 176 Spe
Timid Nature
- Will-O-Wisp
- Thunderbolt
- Ice Punch
- Explosion

Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Hydro Pump
- Ice Beam
- Thunderbolt
- Rapid Spin
"""

class ConstantTeambuilder(Teambuilder):
    def __init__(self, team_string):
        # 1. Parse the showdown string into a list of TeambuilderPokemon
        parsed_team = self.parse_showdown_team(team_string)
        
        # 2. Automatically fix IVs for Hidden Power moves in Gen 3
        fixed_team = fix_gen3_hp_ivs(parsed_team)
        
        # 3. Join the team into the packed format
        self.packed_team = self.join_team(fixed_team)

    def yield_team(self):
        return self.packed_team

async def main():
    teambuilder = ConstantTeambuilder(STAR_TSS_TEAM)

    player_1 = RandomPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        max_concurrent_battles=1
    )
    player_2 = RandomPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        max_concurrent_battles=1
    )

    print("Starting Gen 3 OU battle with Star TSS team...")
    await player_1.battle_against(player_2, n_battles=1)

    print(f"Finished battles: {player_1.n_finished_battles}")
    print(f"Player 1 wins: {player_1.n_won_battles}")

if __name__ == "__main__":
    asyncio.run(main())
