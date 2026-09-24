"""Family L: gen-3 LURES. Six Smogon lure techs, each with a canonical victim (the mon a player naturally
switches in on the lure user). Every set comes from a Smogon sample team (named per lure); the LURE world
and the STANDARD world differ in EXACTLY one move slot (the lure move vs the Smogon-common move for that
slot), with identical EVs/IVs/nature/item, so until the lure move is used the two worlds are
indistinguishable to the viewer (build_states.py ASSERTS the model's obs is byte-identical across them).

Per lure, three states (T = 2; our X leads vs their lure user):
  L<i>_pre_lure  turn 1: X acts, the lure user uses a STANDARD move; its set holds the lure (unrevealed)
  L<i>_pre_std   the same visible prefix; its set holds the standard move instead
  L<i>_post      turn 1: the lure user USES the lure move on X (revealed); lure world
The Smogon prior p = P(lure move | species) comes from data/pokemon/gen3_move_priors.json (read at build).
"""
from __future__ import annotations

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teams as TM  # noqa: E402
from states import StateSpec  # noqa: E402

# ---------------------------------------------------------------- their lure users + teammates
# a3ca232f030ef625.txt: Tyranitar RS/EQ/HP Grass/DD (IVs added so HP is really Grass: 30 Atk / 30 SpA).
TTAR_a3ca = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 176 HP / 108 Atk / 56 SpD / 168 Spe
Naughty Nature
IVs: 31 HP / 30 Atk / 31 Def / 30 SpA / 31 SpD / 31 Spe
- Rock Slide
- Earthquake
- Hidden Power [Grass]
- Dragon Dance"""
A3CA_REST = ["""Zapdos @ Leftovers
Ability: Pressure
EVs: 220 SpA / 48 SpD / 240 Spe
Modest Nature
IVs: 2 Atk / 30 SpA
- Thunderbolt
- Hidden Power [Grass]
- Baton Pass
- Roar""", """Metagross @ Choice Band
Ability: Clear Body
EVs: 128 HP / 252 Atk / 128 Spe
Adamant Nature
- Meteor Mash
- Explosion
- Earthquake
- Rock Slide""", """Celebi @ Leftovers
Ability: Natural Cure
EVs: 208 HP / 84 Def / 36 SpA / 180 Spe
Timid Nature
IVs: 2 Atk / 30 Def
- Calm Mind
- Giga Drain
- Hidden Power [Ice]
- Baton Pass""", """Swampert @ Leftovers
Ability: Torrent
EVs: 252 HP / 136 SpA / 120 Spe
Rash Nature
- Surf
- Substitute
- Focus Punch
- Ice Beam""", """Charizard @ Leftovers
Ability: Blaze
EVs: 4 Atk / 252 SpA / 252 Spe
Mild Nature
IVs: 31 HP / 30 Atk / 31 Def / 30 SpA / 31 SpD / 31 Spe
- Substitute
- Hidden Power [Grass]
- Fire Blast
- Focus Punch"""]

# 9283210847f806ee.txt: Gengar Taunt/WoW/Giga Drain/TB (Modest). MODIFIED for BOTH worlds: Atk IV 0 -> 31
# (a Focus Punch set would not zero Atk); the lure world swaps Taunt -> Focus Punch.
GENGAR_9283 = """Gengar @ Leftovers
Ability: Levitate
EVs: 80 HP / 16 SpA / 236 SpD / 176 Spe
Modest Nature
- Taunt
- Will-O-Wisp
- Giga Drain
- Thunderbolt"""
N9283_REST = ["""Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 80 Atk / 176 Spe
Adamant Nature
IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 30 SpD / 30 Spe
- Dragon Dance
- Hidden Power [Bug]
- Earthquake
- Rock Slide""", """Skarmory @ Leftovers
Ability: Keen Eye
EVs: 248 HP / 244 SpD / 16 Spe
Careful Nature
- Spikes
- Drill Peck
- Protect
- Roar""", """Swampert @ Leftovers
Ability: Torrent
EVs: 252 HP / 168 Def / 88 SpD
Bold Nature
IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Protect
- Surf
- Toxic
- Refresh""", """Zapdos @ Leftovers
Ability: Pressure
EVs: 252 HP / 60 SpA / 196 Spe
Modest Nature
IVs: 31 HP / 2 Atk / 30 Def / 31 SpA / 31 SpD / 31 Spe
- Thunderbolt
- Protect
- Hidden Power [Ice]
- Toxic""", """Aerodactyl @ Choice Band
Ability: Rock Head
EVs: 224 Atk / 32 SpD / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Double-Edge
- Hidden Power [Bug]"""]

# 8e768980fc8f3b5f.txt: Celebi CM/Psychic/Giga Drain/HP Fire; std world: HP Fire -> Recover.
CELEBI_8e76 = """Celebi @ Leftovers
Ability: Natural Cure
EVs: 140 Def / 180 SpA / 188 Spe
Timid Nature
IVs: 31 HP / 2 Atk / 31 Def / 30 SpA / 31 SpD / 30 Spe
- Calm Mind
- Psychic
- Giga Drain
- Hidden Power [Fire]"""
N8E76_REST = ["""Tyranitar @ Lum Berry
Ability: Sand Stream
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Rock Slide
- Earthquake
- Double-Edge""", """Registeel @ Leftovers
Ability: Clear Body
EVs: 252 HP / 56 Def / 112 SpD / 88 Spe
Careful Nature
- Thunder Wave
- Toxic
- Explosion
- Seismic Toss""", """Swampert @ Salac Berry
Ability: Torrent
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Substitute
- Endeavor
- Toxic
- Hydro Pump""", """Raikou @ Leftovers
Ability: Pressure
EVs: 8 Def / 248 SpA / 252 Spe
Timid Nature
IVs: 31 HP / 2 Atk / 31 Def / 30 SpA / 31 SpD / 31 Spe
- Calm Mind
- Thunderbolt
- Crunch
- Hidden Power [Grass]""", """Salamence @ Leftovers
Ability: Intimidate
EVs: 168 Atk / 200 SpD / 140 Spe
Adamant Nature
- Dragon Dance
- Earthquake
- Brick Break
- Rock Slide"""]


def _swap(s: str, old: str, new: str) -> str:
    assert f"- {old}" in s, (old, s)
    return s.replace(f"- {old}", f"- {new}")


def _team(lead: str, rest: List[str]) -> List[str]:
    return [lead] + list(rest)


LURES = [
    # id, lure user (lure world set), std set, their other five, our team (lead X first),
    # T1 (X command, their standard move), lure move command, lure move id, victim species, source note
    dict(lid="L1", species="tyranitar", lure_move="hiddenpowergrass", std_move="crunch", victim="swampert",
         lure_set=TTAR_a3ca, std_set=_swap(TTAR_a3ca, "Hidden Power [Grass]", "Crunch"),
         rest=A3CA_REST, ours=TM.team_D_ours(lead="blissey"),
         x_cmd="move icebeam", std_cmd="move rockslide",
         src="a3ca232f030ef625 Tyranitar HP Grass; victim Swampert (4x)"),
    dict(lid="L2", species="metagross", lure_move="thunderpunch", std_move="meteormash", victim="skarmory",
         lure_set=TM.METAGROSS_8b81, std_set=_swap(TM.METAGROSS_8b81, "Thunder Punch", "Meteor Mash"),
         rest=[s for s in TM.team_D_opp() if s != TM.METAGROSS_8b81], ours=TM.team_D_ours(lead="blissey"),
         x_cmd="move icebeam", std_cmd="move earthquake",
         src="8b81c129de Metagross Thunder Punch; victim Skarmory (2x)"),
    dict(lid="L3", species="gengar", lure_move="focuspunch", std_move="taunt", victim="tyranitar",
         lure_set=_swap(GENGAR_9283, "Taunt", "Focus Punch"), std_set=GENGAR_9283,
         rest=N9283_REST, ours=TM.team_D_ours(lead="blissey"),
         x_cmd="move toxic", std_cmd="move thunderbolt",
         src="9283210847f806ee Gengar (Atk IV 31) Focus Punch; victim Tyranitar (4x)"),
    dict(lid="L4", species="celebi", lure_move="hiddenpowerfire", std_move="recover", victim="skarmory",
         lure_set=CELEBI_8e76, std_set=_swap(CELEBI_8e76, "Hidden Power [Fire]", "Recover"),
         rest=N8E76_REST, ours=TM.team_D_ours(lead="swampert"),
         x_cmd="move icebeam", std_cmd="move psychic",
         src="8e768980fc8f3b5f Celebi HP Fire; victim Skarmory (2x)"),
    dict(lid="L5", species="blissey", lure_move="fireblast", std_move="seismictoss", victim="metagross",
         lure_set=TM.BLISSEY_3650, std_set=_swap(TM.BLISSEY_3650, "Fire Blast", "Seismic Toss"),
         rest=[s for s in TM.team_D_ours() if s not in (TM.BLISSEY_3650,)][:5],
         ours=[TM.SKARMORY_e118, TM.TYRANITAR_e118, TM.BLISSEY_e118, TM.METAGROSS_e118, TM.SALAMENCE_e118,
               TM.CLAYDOL_e118],
         x_cmd="move spikes", std_cmd="move icebeam",
         src="3650f09b2f Blissey Fire Blast; victim Metagross (2x)"),
    dict(lid="L6", species="salamence", lure_move="hydropump", std_move="hiddenpowergrass", victim="tyranitar",
         lure_set=_swap(TM.SALAMENCE_e118, "Hidden Power [Grass]", "Hydro Pump"), std_set=TM.SALAMENCE_e118,
         rest=[s for s in TM.team_B_opp() if s != TM.SALAMENCE_e118], ours=TM.team_D_ours(lead="skarmory"),
         x_cmd="move spikes", std_cmd="move dragonclaw",
         src="e11829f0561ef5a9 Salamence Hydro Pump (the brief's example); victim Tyranitar (2x)"),
]


def lure_states() -> List[StateSpec]:
    out = []
    for L in LURES:
        lure_cmd = "move " + L["lure_move"]
        for kind in ("pre_lure", "pre_std", "post"):
            their = L["std_set"] if kind == "pre_std" else L["lure_set"]
            tcmd = lure_cmd if kind == "post" else L["std_cmd"]
            exp = {"p2_active": L["species"], "p1_first": None}
            exp.pop("p1_first")
            # poke-env records a revealed typed Hidden Power as the GENERIC 'hiddenpower' (the protocol
            # line names no type; only effectiveness can disclose it), so the reveal predicate uses that id.
            seen = "hiddenpower" if L["lure_move"].startswith("hiddenpower") else L["lure_move"]
            if kind == "post":
                exp["p2_moves_include"] = [seen]
            else:
                exp["p2_moves_exclude"] = [seen]
            out.append(StateSpec(
                sid=f"{L['lid']}_{kind}", family="L", our_side="p1",
                p1=list(L["ours"]), p2=_team(their, L["rest"]),
                turns=[([L["x_cmd"]], [tcmd])], T=2, premise=None, expect=exp,
                factors={"lure": L["lid"], "kind": kind, "species": L["species"], "lure_move": L["lure_move"],
                         "std_move": L["std_move"], "victim": L["victim"], "src": L["src"]},
                # the two PRE worlds share one seed, so their visible prefixes are the same battle
                seed_key=f"tactics:{L['lid']}_pre" if kind != "post" else f"tactics:{L['lid']}_post"))
    return out
