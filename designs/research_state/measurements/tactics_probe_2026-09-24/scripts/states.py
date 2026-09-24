"""The constructed decision states. Each state = fixed teams + a SCRIPTED prefix (both sides' choices, turn by
turn) + a fixed sim seed + the probe turn T + which side the MODEL plays + the PREMISE opponent command at T.

A state is only USED after build_states.py has proven it: the snapshot at T must satisfy the state's
``expect`` predicates (HP band, boosts, status, reveals, speed order), otherwise the build REFUSES it.
The seed is searched deterministically (``seed_key:k`` for k = 0, 1, 2, ...) and the first k whose
prefix meets every predicate is registered; the chosen k is recorded, never re-picked after reading.

Families
  DG  model plays the GENGAR side (p1) vs a Dragon Dance Tyranitar (p2)   -- the denial question
  DT  model plays the TYRANITAR side (p2) of the same boards               -- the MIRROR
  B   model plays a Double-Edge Tyranitar (p1) vs Soft-Boiled Blissey (p2) -- recoil-suicide denial
  L   lures (lure.py)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teams as TM  # noqa: E402


@dataclass
class StateSpec:
    sid: str
    family: str
    our_side: str                         # the side the MODEL plays at T
    p1: List[str]                         # showdown-text sets, lead first
    p2: List[str]
    turns: List[Tuple[List[str], List[str]]]   # per turn: (p1 commands, p2 commands) incl. forced switches
    T: int                                # probe turn (the first turn NOT in `turns` for the move round)
    premise: Optional[str]                # the OPPONENT's command at T in the PREMISE regime (None = no premise)
    expect: Dict[str, object] = field(default_factory=dict)
    notes: str = ""
    factors: Dict[str, str] = field(default_factory=dict)
    seed_key: str = ""

    def commands(self) -> List[Tuple[str, str]]:
        out = []
        for c1, c2 in self.turns:
            out += [("p1", c) for c in c1] + [("p2", c) for c in c2]
        return out


# ---------------------------------------------------------------------------------------------- family D
GL_TURNS = {
    # TTar HP level -> Gengar's chip moves (Fire Punch is resisted: ~9-11%; Thunderbolt ~23-27%)
    "high": ["move firepunch"],
    "mid": ["move thunderbolt", "move thunderbolt"],
    "low": ["move thunderbolt", "move thunderbolt", "move thunderbolt", "move thunderbolt"],
}
HP_BANDS = {"high": (0.85, 1.0), "mid": (0.50, 0.72), "low": (0.10, 0.268)}   # low: Explosion (>=104) KOs


def gengar_lead(speed: str, hp: str, *, our_side: str = "p1", pursuit: bool = False) -> StateSpec:
    """GL template: our Gengar leads vs their Tyranitar. Gengar chips TTar; TTar Earthquakes into Levitate
    (no effect, reveals EQ) except on turn 1 of the +1 arms, where it Dragon Dances (reveals DD).

    speed: fast0 (TTar +0, STD Gengar 308 > 213) | slow1 (TTar +1: 319 > STD 308) | fast1 (TTar +1, FAST
    Gengar 335 > 319) | slow2 (TTar +2 after a second Dragon Dance: 426 > 308; family R2, high HP only)."""
    fast_gengar = speed == "fast1"
    boosted = speed in ("slow1", "fast1", "slow2")
    g = GL_TURNS[hp]
    turns = []
    for i, gm in enumerate(g):
        tm = "move dragondance" if (boosted and i == 0) else "move earthquake"
        turns.append(([gm], [tm]))
    if speed == "slow2":        # R2 (owner addition): a second Dragon Dance -> +2 (426 > 308); high HP only
        assert hp == "high", hp
        turns.append((["move firepunch"], ["move dragondance"]))
    T = len(turns) + 1
    fam = "DG" if our_side == "p1" else "DT"
    prem = "move dragondance" if our_side == "p1" else "move explosion"
    sid = f"{fam}_GL_{speed}_{hp}" + ("_pursuit" if pursuit else "")
    return StateSpec(
        sid=sid, family=fam, our_side=our_side,
        p1=TM.team_D_ours(lead="gengar", fast=fast_gengar), p2=TM.team_D_opp(pursuit=pursuit),
        turns=turns, T=T, premise=prem,
        expect={"p1_active": "gengar", "p2_active": "tyranitar",
                "p2_boost_spe": 2 if speed == "slow2" else (1 if boosted else 0),
                "p2_hp_band": HP_BANDS[hp], "p1_status": None, "p2_status": None,
                "p1_first": speed in ("fast0", "fast1")},
        factors={"speed": speed, "ttar_hp": hp, "pursuit": "unrevealed" if pursuit else "absent",
                 "spikes": "none", "swampert": "healthy", "template": "GL"},
        seed_key="tactics:" + sid.replace("DT_", "DG_", 1))


def skarm_lead(variant: str, *, our_side: str = "p1") -> StateSpec:
    """SL template (secondary factors at the FAST, TTar-full-HP corner). Our Skarmory leads vs their TTar;
    turn 1 sets (or not) Spikes on THEIR side and TTar Rock Slides / Pursuits the Skarmory; turn 2 Gengar
    comes in on a Levitate-immune Earthquake. Variants:
      K1  spikes, Pursuit ABSENT (Double-Edge set), TTar Rock Slide
      K2  spikes, Pursuit set UNREVEALED, TTar Rock Slide
      K3  spikes, Pursuit set REVEALED (TTar Pursuits the Skarmory, resisted)
      K4  NO spikes (Skarmory Protects), Pursuit absent, TTar Rock Slide
      K5a Swampert WEAKENED: spikes; Swampert takes an EQ and a Double-Edge before Gengar comes in
      K5b Swampert HEALTHY control for K5a: same turn count; TTar Rock Slides the Swampert (resisted)
    """
    pursuit = variant in ("K2", "K3")
    tt1 = "move pursuit" if variant == "K3" else "move rockslide"
    sk1 = "move protect" if variant == "K4" else "move spikes"
    turns = [([sk1], [tt1])]
    if variant == "K5a":
        turns += [(["switch Swampert"], ["move earthquake"]), (["move icebeam"], ["move doubleedge"])]
    elif variant == "K5b":
        turns += [(["switch Swampert"], ["move rockslide"]), (["move icebeam"], ["move rockslide"])]
    turns += [(["switch Gengar"], ["move earthquake"])]
    T = len(turns) + 1
    fam = "DG" if our_side == "p1" else "DT"
    sid = f"{fam}_SL_{variant}"
    return StateSpec(
        sid=sid, family=fam, our_side=our_side,
        p1=TM.team_D_ours(lead="skarmory"), p2=TM.team_D_opp(pursuit=pursuit),
        turns=turns, T=T, premise="move dragondance" if our_side == "p1" else "move explosion",
        expect={"p1_active": "gengar", "p2_active": "tyranitar", "p2_boost_spe": 0, "p1_status": None,
                "p2_status": None, "p1_first": True,
                "p2_hp_band": (0.70, 1.0) if variant.startswith("K5") else (0.95, 1.0),
                "p2_side_conditions": [] if variant == "K4" else ["spikes"]},
        factors={"speed": "fast0", "ttar_hp": "high",
                 "pursuit": {"K2": "unrevealed", "K3": "revealed"}.get(variant, "absent"),
                 "spikes": "none" if variant == "K4" else "their_side_1",
                 "swampert": "weakened" if variant == "K5a" else "healthy", "template": "SL"},
        seed_key="tactics:" + sid.replace("DT_", "DG_", 1))


# ---------------------------------------------------------------------------------------------- family B
def blissey_state(level: str) -> StateSpec:
    """Our Double-Edge Tyranitar (205 Spe) vs their Soft-Boiled Blissey (153 Spe). Their Salamence leads and
    Dragon Claws our TTar while our Earthquake hits nothing (Flying); Blissey then comes in on an EQ.
    Intimidate leaves our TTar at -1 Atk, so the recoil (1/3 of Double-Edge damage) is smaller than at +0;
    the KO threshold is MEASURED per state (mechanics check), never assumed.
      low   three Dragon Claws  -> TTar below the recoil threshold (Double-Edge suicide denies Soft-Boiled)
      high  one Dragon Claw     -> TTar well above it (Soft-Boiled resolves)
    """
    n_dc = {"low": 3, "high": 1}[level]
    turns = [(["move earthquake"], ["move dragonclaw"]) for _ in range(n_dc)]
    turns += [(["move earthquake"], ["switch Blissey"])]
    T = len(turns) + 1
    opp = [TM.SALAMENCE_e118] + [s for s in TM.team_B_opp() if s != TM.SALAMENCE_e118]
    sid = f"B_{level}"
    return StateSpec(
        sid=sid, family="B", our_side="p1", p1=TM.team_B_ours(), p2=opp, turns=turns, T=T,
        premise="move softboiled",
        expect={"p1_active": "tyranitar", "p2_active": "blissey", "p1_status": None, "p2_status": None,
                "p1_first": True,
                "p1_hp_band": (0.05, 0.16) if level == "low" else (0.35, 0.80),
                "p2_hp_band": (0.45, 0.80)},
        factors={"ttar_hp": level, "template": "B"},
        seed_key=f"tactics:{sid}")


def all_states() -> List[StateSpec]:
    out: List[StateSpec] = []
    for speed in ("fast0", "slow1", "fast1"):
        for hp in ("high", "mid", "low"):
            out.append(gengar_lead(speed, hp))
    for v in ("K1", "K2", "K3", "K4", "K5a", "K5b"):
        out.append(skarm_lead(v))
    # the MIRROR: the model plays Tyranitar on the same boards
    for speed, hp in (("fast0", "low"), ("fast0", "high"), ("fast1", "low"), ("slow1", "low")):
        out.append(gengar_lead(speed, hp, our_side="p2"))
    for lv in ("low", "high"):
        out.append(blissey_state(lv))
    # family R (owner addition, amendment 1): R2's +2 board joins the GL core; R1/R3 live in r_family.py
    out.append(gengar_lead("slow2", "high"))
    import r_family  # noqa: E402
    out += r_family.r_states()
    try:
        import lure  # noqa: E402
        out += lure.lure_states()
    except ImportError:
        pass
    return out


def by_id() -> Dict[str, StateSpec]:
    return {s.sid: s for s in all_states()}
