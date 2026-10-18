"""Teams posted INLINE (no paste link), TRANSCRIBED from the public page through the WebFetch reader.

FINDING F2 (README): the reader is a summarising model, so each row was read TWICE with differently
worded prompts (a verbatim block, then a per-field list) and kept only when both reads agreed on
species, item, nature, EVs, IVs and moves. ``transcribed: True`` marks them in the manifest.
"""

INLINE = [
    dict(source_url="https://www.smogon.com/forums/threads/steelwork.3609872/",
         thread_url="https://www.smogon.com/forums/threads/steelwork.3609872/",
         thread_title="Gen 3 - Steelwork", team_name="Steelwork (Jirachi + sand TSS, Aerodactyl)",
         credited="Foggi", post_author="Foggi", post_date="2017-07-19", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Tyranitar @ Leftovers
Ability: Sand Stream
Shiny: Yes
EVs: 240 HP / 12 Atk / 176 SpA / 48 SpD / 32 Spe
Quiet Nature
- Crunch
- Pursuit
- Fire Blast
- Rock Slide

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 248 HP / 184 Def / 24 SpA / 32 SpD / 20 Spe
Bold Nature
IVs: 0 Atk
- Wish
- Protect
- Toxic
- Fire Punch

Skarmory @ Leftovers
Ability: Keen Eye
Shiny: Yes
EVs: 252 HP / 244 SpD / 12 Spe
Careful Nature
IVs: 0 Atk
- Spikes
- Whirlwind
- Protect
- Toxic

Flygon @ Leftovers
Ability: Levitate
EVs: 96 HP / 252 Atk / 20 SpD / 140 Spe
Adamant Nature
- Toxic
- Fire Blast
- Rock Slide
- Earthquake

Snorlax @ Leftovers
Ability: Immunity
EVs: 112 HP / 76 Atk / 176 Def / 64 SpA / 80 SpD
Sassy Nature
- Body Slam
- Fire Blast
- Earthquake
- Self-Destruct

Aerodactyl @ Choice Band
Ability: Rock Head
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Double-Edge
- Hidden Power [Bug]
- Rock Slide
- Earthquake
"""),
    dict(source_url="https://www.smogon.com/forums/threads/komm-süßer-tod-a-guide-to-venupert.3611254/",
         thread_url="https://www.smogon.com/forums/threads/komm-süßer-tod-a-guide-to-venupert.3611254/",
         thread_title="Gen 3 - Komm, süßer Tod (A guide to VenuPert)", team_name="VenuPert (Venusaur + Swampert Spikes + Aerodactyl)",
         credited="eden", post_author="eden", post_date="2017-08-04", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Sleep in Sin (Venusaur) (M) @ Leftovers
Ability: Overgrow
EVs: 180 HP / 28 Atk / 80 Def / 108 SpD / 112 Spe
Calm Nature
- Sleep Powder
- Sludge Bomb
- Leech Seed
- Giga Drain

Sayonara Sunrise (Tyranitar) (M) @ Leftovers
Ability: Sand Stream
EVs: 240 HP / 12 Def / 252 SpA / 4 Spe
Modest Nature
IVs: 2 Atk / 30 SpA
- Crunch
- Pursuit
- Hidden Power [Grass]
- Flamethrower

Twisting Fate (Forretress) (M) @ Leftovers
Ability: Sturdy
EVs: 252 HP / 4 Atk / 252 SpD
Sassy Nature
- Spikes
- Rapid Spin
- Explosion
- Earthquake

Broken Dreams (Aerodactyl) (M) @ Choice Band
Ability: Rock Head
EVs: 224 Atk / 32 SpD / 252 Spe
Jolly Nature
- Rock Slide
- Earthquake
- Hidden Power [Bug]
- Double-Edge

Heaven's Herald (Zapdos) @ Leftovers
Ability: Pressure
EVs: 248 HP / 36 Def / 164 SpD / 60 Spe
Calm Nature
IVs: 2 Atk / 30 SpA
- Thunderbolt
- Toxic
- Thunder Wave
- Hidden Power [Grass]

Riptide Resonance (Swampert) (M) @ Leftovers
Ability: Torrent
EVs: 240 HP / 176 Def / 44 SpD / 48 Spe
Relaxed Nature
- Earthquake
- Ice Beam
- Rest
- Protect
"""),
    dict(source_url="https://www.smogon.com/forums/threads/late-game-zapdos-team-analysis.3578828/",
         thread_url="https://www.smogon.com/forums/threads/late-game-zapdos-team-analysis.3578828/",
         thread_title="Gen 3 - Late Game Zapdos - Team Analysis", team_name="Late Game Zapdos (Agility Pass Zapdos)",
         credited="Astamatitos", post_author="Astamatitos", post_date="2016-08-07", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Tyranitar @ Choice Band
Ability: Sand Stream
EVs: 80 HP / 188 Atk / 80 Def / 160 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Hidden Power [Bug]
- Focus Punch

Zapdos @ Leftovers
Ability: Pressure
EVs: 112 HP / 252 SpA / 144 Spe
Modest Nature
IVs: 2 Atk / 30 SpA
- Thunderbolt
- Hidden Power [Grass]
- Agility
- Baton Pass

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 Def / 180 SpA / 76 Spe
Bold Nature
IVs: 0 Atk
- Seismic Toss
- Ice Beam
- Aromatherapy
- Soft-Boiled

Dugtrio @ Choice Band
Ability: Arena Trap
EVs: 8 HP / 228 Atk / 40 SpD / 232 Spe
Adamant Nature
IVs: 30 SpD / 30 Spe
- Earthquake
- Rock Slide
- Aerial Ace
- Hidden Power [Bug]

Swampert @ Salac Berry
Ability: Torrent
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Surf
- Substitute
- Endeavor
- Roar

Claydol @ Leftovers
Ability: Levitate
EVs: 248 HP / 196 Def / 24 SpD / 40 Spe
Sassy Nature
- Earthquake
- Psychic
- Rapid Spin
- Explosion
"""),
    dict(source_url="https://www.smogon.com/forums/threads/explosion-is-an-art.3583380/",
         thread_url="https://www.smogon.com/forums/threads/explosion-is-an-art.3583380/",
         thread_title="OU Explosion is an art", team_name="Explosion is an art (Boom Spam)",
         credited="Foggi", post_author="Foggi", post_date="2016-09-27", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Metagross @ Leftovers
Ability: Clear Body
EVs: 208 HP / 252 SpA / 48 SpD
Quiet Nature
- Meteor Mash
- Psychic
- Hidden Power [Fire]
- Explosion

Cloyster (M) @ Leftovers
Ability: Shell Armor
EVs: 252 HP / 40 SpA / 204 SpD / 12 Spe
Relaxed Nature
- Explosion
- Ice Beam
- Rapid Spin
- Spikes

Gengar @ Leftovers
Ability: Levitate
EVs: 112 HP / 84 Atk / 124 Def / 188 Spe
Naive Nature
- Thunderbolt
- Ice Punch
- Hypnosis
- Explosion

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 80 SpA / 176 Spe
Modest Nature
IVs: 0 Atk
- Calm Mind
- Surf
- Substitute
- Roar

Snorlax @ Leftovers
Ability: Immunity
EVs: 96 HP / 56 Atk / 136 Def / 80 SpA / 140 SpD
Brave Nature
- Body Slam
- Brick Break
- Self-Destruct
- Fire Blast

Camerupt @ Leftovers
Ability: Magma Armor
EVs: 252 HP / 20 Atk / 124 SpA / 112 SpD
Quiet Nature
- Fire Blast
- Earthquake
- Toxic
- Protect
"""),
    dict(source_url="https://www.smogon.com/forums/threads/ozymandias.3533733/",
         thread_url="https://www.smogon.com/forums/threads/ozymandias.3533733/",
         thread_title="Gen 3 - Ozymandias", team_name="Ozymandias (Smurf Rain)",
         credited="Smurf.", post_author="Smurf.", post_date="2015-03-27", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Metagross @ Choice Band
Ability: Clear Body
EVs: 64 HP / 252 Atk / 192 Spe
Adamant Nature
- Earthquake
- Meteor Mash
- Explosion
- Rock Slide

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 76 Atk / 180 Spe
Adamant Nature
- Hidden Power [Bug]
- Dragon Dance
- Earthquake
- Rock Slide

Snorlax @ Lum Berry
Ability: Thick Fat
EVs: 160 HP / 188 Atk / 120 Def / 40 SpD
Adamant Nature
- Counter
- Shadow Ball
- Curse
- Self-Destruct

Gengar @ Leftovers
Ability: Levitate
EVs: 124 Atk / 200 SpA / 184 Spe
Hasty Nature
- Thunderbolt
- Explosion
- Ice Punch
- Will-O-Wisp

Kingdra @ Leftovers
Ability: Swift Swim
EVs: 160 HP / 252 SpA / 96 Spe
Modest Nature
- Rain Dance
- Hydro Pump
- Ice Beam
- Hidden Power [Electric]

Swampert @ Salac Berry
Ability: Torrent
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Substitute
- Hydro Pump
- Ice Beam
- Endeavor
"""),
    dict(source_url="https://www.smogon.com/forums/threads/querencia.3521102/",
         thread_url="https://www.smogon.com/forums/threads/querencia.3521102/",
         thread_title="Gen 3 querencia", team_name="querencia (OffStar Mag)",
         credited="McMeghan", post_author="McMeghan", post_date="2014-11-01", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Tyranitar (M) @ Choice Band
Trait: Sand Stream
EVs: 48 HP / 252 Atk / 208 Spd
Adamant Nature (+Atk, -SAtk)
- Rock Slide
- Earthquake
- Focus Punch
- Hidden Power [Bug]

Starmie @ Leftovers
Trait: Natural Cure
EVs: 4 HP / 252 SAtk / 252 Spd
Timid Nature (+Spd, -Atk)
- Hydro Pump
- Thunderbolt
- Ice Beam
- Recover

Gengar (F) @ Leftovers
Trait: Levitate
EVs: 124 Atk / 200 SAtk / 184 Spd
Hasty Nature (+Spd, -Def)
- Thunderbolt
- Explosion
- Ice Punch
- Giga Drain

Swampert (F) @ Leftovers
Trait: Torrent
EVs: 252 HP / 224 Def / 32 Spd
Relaxed Nature (+Def, -Spd)
- Rest
- Protect
- Ice Beam
- Earthquake

Magneton @ Leftovers
Trait: Magnet Pull
EVs: 112 HP / 252 SAtk / 44 SDef / 100 Spd
Modest Nature (+SAtk, -Atk)
- Thunderbolt
- Hidden Power [Fire]
- Sleep Talk
- Rest

Flygon (M) @ Choice Band
Trait: Levitate
EVs: 4 HP / 252 Atk / 252 Spd
Jolly Nature (+Spd, -SAtk)
- Earthquake
- Rock Slide
- Hidden Power [Flying]
- Toxic
"""),
    dict(source_url="https://www.smogon.com/forums/threads/special-offense-spikes.3632230/",
         thread_url="https://www.smogon.com/forums/threads/special-offense-spikes.3632230/",
         thread_title="Gen 3 Special Offense + Spikes", team_name="Special Offense + Spikes",
         credited="UD", post_author="UD", post_date="2018-04-10", tier="A",
         category="resource_listed_team_analysis", transcribed=True,
         text="""Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 248 HP / 252 Atk / 4 SpD / 4 Spe
Adamant Nature
- Earthquake
- Focus Punch
- Hidden Power [Bug]
- Rock Slide

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Drill Peck
- Spikes
- Toxic
- Roar

Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 2 Atk / 30 Def
- Baton Pass
- Hidden Power [Ice]
- Roar
- Thunderbolt

Gengar @ Leftovers
Ability: Levitate
EVs: 40 Atk / 252 SpA / 216 Spe
Hasty Nature
IVs: 30 Def / 30 SpA
- Dynamic Punch
- Hidden Power [Grass]
- Ice Punch
- Fire Punch

Zapdos @ Leftovers
Ability: Pressure
EVs: 228 Atk / 140 SpA / 140 Spe
Mild Nature
- Agility
- Drill Peck
- Hidden Power [Grass]
- Thunderbolt

Flygon @ Leftovers
Ability: Levitate
EVs: 208 HP / 136 Atk / 144 Def / 20 Spe
Adamant Nature
- Earthquake
- Protect
- Rock Slide
- Toxic
"""),
]
