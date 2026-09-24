"""Family R (ROLES), the owner's 2026-09-24 addition: does the policy / critic PRICE a role, and know that a
role's value depends on the opponent's roster?

R1 "Aerodactyl is a sweeper" -- their Aerodactyl is the 9283210847f806ee sample set verbatim (Choice Band,
   Jolly 224 Atk / 32 SpD / 252 Spe: Earthquake / Rock Slide / Double-Edge / Hidden Power Bug; a Smogon
   sample, so gen-3 legal). Our side is the SkarmBliss sample (3650f09b2f), whose Skarmory and Swampert
   check it.
   R1_entry                 T=1, our Swampert vs their Aerodactyl lead: the BELIEF read (moves, item, spread)
   R1_{alive,fainted}_sk{healthy,weak}  T=4, our Skarmory vs their Zapdos. Aerodactyl ALIVE (it retreated on
                            turn 1) or FAINTED (Hydro Pump KO on turn 1); Skarmory HEALTHY (Zapdos Toxics
                            into Steel) or nearly GONE (Zapdos Thunderbolts it). Zapdos took exactly one
                            Hydro Pump in every arm. The role contrast is the INTERACTION: losing the
                            check should cost more when the thing it checks is alive.
R2 "boosted Tyranitar" -- the DG_GL high-HP boards at +0 / +1 / +2 (DG_GL_fast0_high, DG_GL_slow1_high and
   the new DG_GL_slow2_high, defined in states.py).
R3 "Pursuit Tyranitar vs the opposing Gengar alive / fainted" -- OUR Tyranitar is the e541f7be8713393c
   sample's special Pursuit set (Crunch / Pursuit / Fire Blast / Brick Break); their team is SkarmBliss.
   R3_pursuit               T=1, our Tyranitar vs their Gengar lead: Pursuit vs the rest
   R3_{alive,fainted}       T=3, our fresh Tyranitar vs their Swampert. Their Gengar retreated from our
                            Choice Band Metagross on turn 1 (ALIVE) or stayed and was Meteor Mashed (FAINTED)
   R3m_*                    the MIRROR, the model on the SkarmBliss side reading P(Pursuit) on OUR
                            Tyranitar: at T=1 with its Gengar vs with its Blissey in front (lead), and at
                            T=2 after the Tyranitar SWITCHED IN on its Gengar vs on its Blissey.
"""
from __future__ import annotations

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teams as TM  # noqa: E402
import lure as LU  # noqa: E402
from states import StateSpec  # noqa: E402

AERODACTYL_9283 = """Aerodactyl @ Choice Band
Ability: Rock Head
EVs: 224 Atk / 32 SpD / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Double-Edge
- Hidden Power [Bug]"""
ZAPDOS_9283 = LU.N9283_REST[3]
TTAR_9283, SKARM_9283, SWAMP_9283 = LU.N9283_REST[0], LU.N9283_REST[1], LU.N9283_REST[2]
GENGAR_9283_SAMPLE = LU.GENGAR_9283.replace("Modest Nature\n", "Modest Nature\nIVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe\n")

# e541f7be8713393c.txt
E541 = {
    "metagross": """Metagross @ Choice Band
Ability: Clear Body
EVs: 252 Atk / 12 Def / 244 Spe
Jolly Nature
- Meteor Mash
- Earthquake
- Explosion
- Double-Edge""",
    "snorlax": """Snorlax @ Leftovers
Ability: Immunity
EVs: 24 HP / 176 Atk / 108 Def / 184 SpD / 16 Spe
Adamant Nature
- Body Slam
- Focus Punch
- Earthquake
- Self-Destruct""",
    "suicune": """Suicune @ Leftovers
Ability: Pressure
EVs: 228 HP / 212 SpA / 68 Spe
Modest Nature
IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Calm Mind
- Surf
- Rest
- Sleep Talk""",
    "claydol": """Claydol @ Leftovers
Ability: Levitate
EVs: 112 HP / 240 Atk / 8 Def / 148 Spe
Adamant Nature
- Earthquake
- Explosion
- Rapid Spin
- Rock Slide""",
    "tyranitar": """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 184 HP / 244 SpA / 80 Spe
Quiet Nature
- Crunch
- Pursuit
- Fire Blast
- Brick Break""",
    "salamence": """Salamence @ Leftovers
Ability: Intimidate
EVs: 116 HP / 252 Atk / 140 Spe
Adamant Nature
IVs: 30 SpA / 30 SpD / 30 Spe
- Dragon Dance
- Hidden Power [Flying]
- Rock Slide
- Earthquake""",
}


def e541(lead: str) -> List[str]:
    order = [lead] + [k for k in ("metagross", "snorlax", "suicune", "claydol", "tyranitar", "salamence") if k != lead]
    return [E541[k] for k in order]


def r_states() -> List[StateSpec]:
    out: List[StateSpec] = []
    # ---------------------------------------------------------------- R1
    theirs_aero = [AERODACTYL_9283, ZAPDOS_9283, TTAR_9283, SKARM_9283, SWAMP_9283, GENGAR_9283_SAMPLE]
    ours_swamp = TM.team_D_ours(lead="swampert")
    out.append(StateSpec(
        sid="R1_entry", family="R", our_side="p1", p1=ours_swamp, p2=theirs_aero, turns=[], T=1, premise=None,
        expect={"p1_active": "swampert", "p2_active": "aerodactyl"},
        factors={"role": "R1", "kind": "belief_entry"}, seed_key="tactics:R1_entry"))
    for aero in ("alive", "fainted"):
        for sk in ("healthy", "weak"):
            if aero == "fainted":
                turns = [(["move hydropump"], ["move rockslide", "switch Zapdos"]),
                         (["move hydropump"], ["move hiddenpowerice"])]
            else:
                turns = [(["move hydropump"], ["switch Zapdos"]),
                         (["move protect"], ["move hiddenpowerice"])]
            turns.append((["switch Skarmory"], ["move toxic" if sk == "healthy" else "move thunderbolt"]))
            exp = {"p1_active": "skarmory", "p2_active": "zapdos", "p1_status": None, "p2_status": None,
                   "p1_hp_band": (0.95, 1.0) if sk == "healthy" else (0.05, 0.35),
                   "p2_hp_band": (0.55, 0.80),
                   "p2_team_hp": {"aerodactyl": (0.0, 0.0) if aero == "fainted" else (1.0, 1.0)}}
            out.append(StateSpec(
                sid=f"R1_{aero}_sk{sk}", family="R", our_side="p1", p1=ours_swamp, p2=theirs_aero,
                turns=turns, T=4, premise=None, expect=exp,
                factors={"role": "R1", "aero": aero, "skarmory": sk},
                seed_key=f"tactics:R1_{aero}_sk{sk}"))
    # ---------------------------------------------------------------- R3
    ss_gengar = TM.team_D_ours(lead="gengar")          # SkarmBliss, Gengar lead
    ss_bliss = TM.team_D_ours(lead="blissey")
    out.append(StateSpec(
        sid="R3_pursuit", family="R", our_side="p1", p1=e541("tyranitar"), p2=ss_gengar, turns=[], T=1,
        premise=None, expect={"p1_active": "tyranitar", "p2_active": "gengar"},
        factors={"role": "R3", "kind": "pursuit_vs_gengar"}, seed_key="tactics:R3_pursuit"))
    for g in ("alive", "fainted"):
        t1 = ["move thunderbolt", "switch Swampert"] if g == "fainted" else ["switch Swampert"]
        turns = [(["move meteormash"], t1), (["switch Tyranitar"], ["move protect"])]
        out.append(StateSpec(
            sid=f"R3_{g}", family="R", our_side="p1", p1=e541("metagross"), p2=ss_gengar, turns=turns, T=3,
            premise=None,
            expect={"p1_active": "tyranitar", "p2_active": "swampert", "p1_hp_band": (0.95, 1.0),
                    "p1_status": None, "p2_status": None,
                    "p2_team_hp": {"gengar": (0.0, 0.0) if g == "fainted" else (1.0, 1.0)}},
            factors={"role": "R3", "gengar": g}, seed_key=f"tactics:R3_{g}"))
    # the MIRROR (model = the SkarmBliss side, p2), belief on OUR Pursuit Tyranitar
    for front, team in (("gengar", ss_gengar), ("blissey", ss_bliss)):
        out.append(StateSpec(
            sid=f"R3m_lead_{front}", family="R", our_side="p2", p1=e541("tyranitar"), p2=team, turns=[], T=1,
            premise=None, expect={"p1_active": "tyranitar", "p2_active": front},
            factors={"role": "R3m", "kind": "lead", "front": front}, seed_key=f"tactics:R3m_lead_{front}"))
        tmove = "move thunderbolt" if front == "gengar" else "move icebeam"
        out.append(StateSpec(
            sid=f"R3m_swin_{front}", family="R", our_side="p2", p1=e541("metagross"), p2=team,
            turns=[(["switch Tyranitar"], [tmove])], T=2, premise=None,
            expect={"p1_active": "tyranitar", "p2_active": front},
            factors={"role": "R3m", "kind": "switch_in", "front": front}, seed_key=f"tactics:R3m_swin_{front}"))
    return out
