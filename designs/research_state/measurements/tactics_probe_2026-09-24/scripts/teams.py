"""Fixture teams. Every set is copied from a Smogon SAMPLE team under data/teams/sample/ (the file is
named on each block); a MODIFIED set says what changed and why. Lead = first element of each list.

Speeds (level 100, verified in-sim by verify_rule.py / build_states.py, never assumed):
  Gengar STD  (248 HP/52 Def/12 SpA/100 SpD/96 Spe Timid)  -> 308
  Gengar FAST (248 HP/52 Def/12 SpA/196 Spe Timid; SpD EVs moved to Spe) -> 335
  Tyranitar DD (184 HP/104 Atk/220 Spe Adamant) -> 213, x1.5 at +1 = 319
So: +0 TTar is slower than both Gengars; +1 TTar outspeeds STD Gengar (319 > 308) and is outsped by
FAST Gengar (335 > 319). Holding TTar at +1 and swapping only our Gengar's Spe/SpD EVs isolates speed order.
"""

# ---------------------------------------------------------------- our side for family D (the Gengar side)
# 3650f09b2f.txt (SkarmBliss). Order changed only (lead chosen per template).
SKARMORY_3650 = """Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 8 Def / 248 SpD
Careful Nature
- Spikes
- Protect
- Whirlwind
- Toxic"""
BLISSEY_3650 = """Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 Def / 112 SpA / 144 Spe
Modest Nature
IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Soft-Boiled
- Ice Beam
- Toxic
- Fire Blast"""
TYRANITAR_3650 = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 248 HP / 196 Atk / 12 Def / 52 SpD
Adamant Nature
IVs: 31 HP / 30 Atk / 30 Def / 31 SpA / 30 SpD / 31 Spe
- Earthquake
- Rock Slide
- Hidden Power Bug
- Roar"""
SWAMPERT_3650 = """Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 136 Def / 40 SpA / 48 SpD / 44 Spe
Relaxed Nature
- Earthquake
- Ice Beam
- Hydro Pump
- Protect"""
GENGAR_3650 = """Gengar @ Leftovers
Ability: Levitate
EVs: 248 HP / 52 Def / 12 SpA / 100 SpD / 96 Spe
Timid Nature
- Fire Punch
- Thunderbolt
- Explosion
- Will-O-Wisp"""
# MODIFIED: the 100 SpD EVs moved to Spe (96 -> 196 Spe). Everything the opponent's physical attacks
# see (HP, Def) is unchanged; only Spe and SpD differ from GENGAR_3650.
GENGAR_3650_FAST = GENGAR_3650.replace("100 SpD / 96 Spe", "196 Spe")
STARMIE_3650 = """Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Hydro Pump
- Ice Beam
- Thunderbolt
- Rapid Spin"""

# ---------------------------------------------------------------- opponent for family D (the Tyranitar side)
# 8b81c129de.txt. Order changed only.
ZAPDOS_8b81 = """Zapdos @ Leftovers
Ability: Pressure
EVs: 216 HP / 136 SpA / 156 Spe
Adamant Nature
IVs: 30 Spe
- Thunder Wave
- Toxic
- Thunderbolt
- Hidden Power [Ice]"""
SWAMPERT_8b81 = """Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 252 SpA / 16 Spe
Quiet Nature
- Curse
- Earthquake
- Hydro Pump
- Ice Beam"""
METAGROSS_8b81 = """Metagross @ Leftovers
Ability: Clear Body
EVs: 168 HP / 16 Def / 252 SpA / 72 Spe
Rash Nature
- Earthquake
- Thunder Punch
- Hidden Power [Grass]
- Explosion"""
TYRANITAR_8b81 = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 184 HP / 104 Atk / 220 Spe
Adamant Nature
- Dragon Dance
- Earthquake
- Rock Slide
- Double-Edge"""
# MODIFIED: Double-Edge -> Pursuit (Pursuit is 0.134 of Smogon gen-3 Tyranitar usage; DD+Pursuit is legal:
# Pursuit by level-up, Dragon Dance by egg move).
TYRANITAR_8b81_PURSUIT = TYRANITAR_8b81.replace("- Double-Edge", "- Pursuit")
SNORLAX_8b81 = """Snorlax @ Leftovers
Ability: Immunity
EVs: 32 HP / 136 Atk / 152 Def / 172 SpD / 16 Spe
Adamant Nature
- Body Slam
- Shadow Ball
- Focus Punch
- Self-Destruct"""
SALAMENCE_8b81 = """Salamence @ Leftovers
Ability: Intimidate
EVs: 156 Atk / 212 SpD / 140 Spe
Adamant Nature
IVs: 30 SpA / 30 SpD / 30 Spe
- Dragon Dance
- Earthquake
- Rock Slide
- Hidden Power [Flying]"""


def team_D_ours(lead: str = "gengar", fast: bool = False):
    g = GENGAR_3650_FAST if fast else GENGAR_3650
    rest = {"gengar": g, "skarmory": SKARMORY_3650, "swampert": SWAMPERT_3650,
            "blissey": BLISSEY_3650, "tyranitar": TYRANITAR_3650, "starmie": STARMIE_3650}
    order = [lead] + [k for k in ("skarmory", "gengar", "swampert", "blissey", "tyranitar", "starmie") if k != lead]
    return [rest[k] for k in order]


def team_D_opp(pursuit: bool = False):
    t = TYRANITAR_8b81_PURSUIT if pursuit else TYRANITAR_8b81
    return [t, ZAPDOS_8b81, SWAMPERT_8b81, METAGROSS_8b81, SNORLAX_8b81, SALAMENCE_8b81]


# ---------------------------------------------------------------- family B (Double-Edge recoil vs Soft-Boiled)
# our side: 710d8d529538ff90.txt (the DD/RS/EQ/Double-Edge Tyranitar); order changed only.
ZAPDOS_710d = """Zapdos @ Leftovers
Ability: Pressure
EVs: 240 HP / 120 Atk / 148 Spe
Hasty Nature
- Drill Peck
- Hidden Power [Grass]
- Baton Pass
- Thunderbolt"""
METAGROSS_710d = """Metagross @ Choice Band
Ability: Clear Body
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Meteor Mash
- Double-Edge
- Earthquake
- Explosion"""
TYRANITAR_710d = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 184 HP / 112 Atk / 24 Def / 188 Spe
Adamant Nature
- Dragon Dance
- Rock Slide
- Earthquake
- Double-Edge"""
SWAMPERT_710d = """Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 44 Def / 180 SpA / 44 Spe
Quiet Nature
- Earthquake
- Focus Punch
- Ice Beam
- Hydro Pump"""
JIRACHI_710d = """Jirachi @ Leftovers
Ability: Serene Grace
EVs: 236 HP / 52 Def / 180 SpA / 40 Spe
Modest Nature
IVs: 2 Atk / 30 SpA
- Calm Mind
- Psychic
- Fire Punch
- Hidden Power [Grass]"""
SALAMENCE_710d = """Salamence @ Leftovers
Ability: Intimidate
EVs: 116 HP / 252 Atk / 140 Spe
Adamant Nature
IVs: 30 SpA / 30 SpD / 30 Spe
- Dragon Dance
- Hidden Power [Flying]
- Rock Slide
- Earthquake"""

# opponent: e11829f0561ef5a9.txt (Blissey with Soft-Boiled); order changed only.
TYRANITAR_e118 = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 184 HP / 116 Atk / 20 Def / 188 Spe
Adamant Nature
- Dragon Dance
- Rock Slide
- Earthquake
- Hidden Power [Bug]"""
SKARMORY_e118 = """Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 SpD / 4 Spe
Careful Nature
- Spikes
- Roar
- Protect
- Drill Peck"""
BLISSEY_e118 = """Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 Def / 228 SpA / 28 Spe
Modest Nature
IVs: 0 Atk
- Thunderbolt
- Calm Mind
- Ice Beam
- Soft-Boiled"""
METAGROSS_e118 = """Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 56 Atk / 140 Def / 60 Spe
Adamant Nature
- Explosion
- Protect
- Meteor Mash
- Earthquake"""
SALAMENCE_e118 = """Salamence @ Leftovers
Ability: Intimidate
EVs: 4 Atk / 252 SpA / 252 Spe
Mild Nature
IVs: 30 HP / 30 SpA
- Dragon Claw
- Brick Break
- Hidden Power [Grass]
- Flamethrower"""
CLAYDOL_e118 = """Claydol @ Leftovers
Ability: Levitate
EVs: 252 HP / 196 Atk / 60 Spe
Adamant Nature
- Earthquake
- Psychic
- Explosion
- Rapid Spin"""


def team_B_ours():
    return [TYRANITAR_710d, ZAPDOS_710d, METAGROSS_710d, SWAMPERT_710d, JIRACHI_710d, SALAMENCE_710d]


def team_B_opp():
    return [BLISSEY_e118, TYRANITAR_e118, SKARMORY_e118, METAGROSS_e118, SALAMENCE_e118, CLAYDOL_e118]
