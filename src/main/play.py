import argparse
import asyncio
from poke_env.player import RandomPlayer
from utils.teambuilder import Gen3Teambuilder

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

async def main(proxy_url=None):
    teambuilder = Gen3Teambuilder(STAR_TSS_TEAM)

    player_1 = RandomPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        max_concurrent_battles=1,
        proxy_url=proxy_url,
    )
    player_2 = RandomPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        max_concurrent_battles=1,
        proxy_url=proxy_url,
    )

    print(f"Starting Gen 3 OU battle (proxy={'enabled' if proxy_url else 'disabled'})...")
    await player_1.battle_against(player_2, n_battles=1)

    print(f"Finished battles: {player_1.n_finished_battles}")
    print(f"Player 1 wins: {player_1.n_won_battles}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play on Pokemon Showdown")
    parser.add_argument(
        "--proxy", type=str, default=None, metavar="SOCKS5_URL",
        help="SOCKS5 proxy URL, e.g. socks5h://127.0.0.1:1080",
    )
    args = parser.parse_args()
    asyncio.run(main(proxy_url=args.proxy))
