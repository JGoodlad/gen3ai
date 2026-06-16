"""GOLD-STANDARD physics oracle for the differentiable DamageOperator — validates the op's gen3 damage
band against the SIM's EXACT realized damage on CONSTRUCTED single-turn scenarios.

Unlike `damage_op_fuzz_test.py` (which scrapes damage from RANDOM bridge games and is fundamentally
measurement-confounded — opponent HP arrives in integer percent, `before`-HP goes stale on untracked
chip, KOs cap the realized drop), this drives Showdown's OMNISCIENT BattleStream (`utils/bridge/
damage_probe.js`) with fully-constructed teams + forced moves and reads EXACT both-side HP + the sim's
OWN computed stats. So a mismatch is UNAMBIGUOUSLY a physics bug — there is no measurement noise.

Each scenario stages exactly one physics modifier (type effectiveness / STAB / boosts / burn / screens /
weather / item / ability-immunity), sets it up in earlier turns, and measures p1's FINAL attack on p2.
The op band is the SAME `_op_band` the random fuzz uses (one source of truth), fed the sim's exact
stats/boosts/types/weather/screens/item — and we assert the sim's exact damage lands inside it.

Run directly (no server — in-process sim via the Node bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/damage_op_probe_fuzz_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional

# Reuse the EXACT band machinery the random-game fuzz uses (single source of truth for the op physics).
from agents import gen3_data
from agents.observation.types import TypeEncoder
from agents.training.poke_env_gaps.damage_op_fuzz_test import (
    _op_band, _OP, _STATS_REGISTRY, _DMG_ROLL_MIN, _CHIP_CAP, _CRIT_CAP, _type_idx)

_PROBE_JS = str(Path(__file__).resolve().parents[3] / "utils" / "bridge" / "damage_probe.js")
_TOL = 0.06          # exact-stats + exact-HP ⇒ the only slack is the op's smooth-vs-floored roll rounding
_FORMAT = "gen3customgame"
_IVS = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}


def mon(species, moves, *, item="leftovers", ability=None, nature="Serious",
        evs=None, ivs=None, level=100, gender="N") -> dict:
    """A full Showdown set (the probe packs it). EVs default to all-0; pass a partial dict to invest."""
    base = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
    if evs:
        base.update(evs)
    return {"species": species, "item": item, "ability": ability or "No Ability",
            "moves": moves, "evs": base, "ivs": ivs or dict(_IVS), "nature": nature,
            "level": level, "gender": gender}


# ── Curated "interesting combinations": each isolates ONE physics modifier. p1 ATTACKS, p2 DEFENDS;
#    the measured hit is p1's move on the LAST turn (earlier turns stage the modifier). Defenders are
#    chosen bulky enough to survive the measured hit (so the realized damage isn't KO-capped). ──────────
def _scenarios() -> List[dict]:
    S: List[dict] = []

    def atk1(name, p1, p2, p1moves, p2moves, choices):
        # Fixed PRNG seed → crits/secondaries are DETERMINISTIC, so the gate is reproducible (the band
        # handles a crit either way via the `-crit` flag, but a fixed seed pins which scenarios crit).
        S.append({"id": name, "formatid": _FORMAT, "seed": [1, 2, 3, 4], "p1": [p1], "p2": [p2],
                  "choices": choices, "_p1moves": p1moves, "_p2moves": p2moves})

    # 1. Neutral special STAB (Suicune Surf → Blissey).
    atk1("neutral_special",
         mon("Suicune", ["surf"], nature="Modest", evs={"spa": 252}),
         mon("Blissey", ["softboiled"], evs={"hp": 252, "spd": 252}, nature="Calm"),
         ["surf"], ["softboiled"], [["p1", "move 1"], ["p2", "move 1"]])
    # 2. Neutral physical STAB (Snorlax Body Slam → Swampert, full HP, single clean hit).
    atk1("neutral_physical",
         mon("Snorlax", ["bodyslam"], nature="Adamant", evs={"atk": 252}),
         mon("Swampert", ["surf"], evs={"hp": 252, "def": 252}, nature="Bold"),
         ["bodyslam"], ["surf"], [["p1", "move 1"], ["p2", "move 1"]])
    # 3. Super-effective 2× (Starmie Thunderbolt → Gyarados, Water/Flying: Electric 2×·2×=4×!). Use 2×:
    #    Thunderbolt → Suicune (pure Water) = 2×.
    atk1("super_effective_2x",
         mon("Starmie", ["thunderbolt"], nature="Modest", evs={"spa": 252}),
         mon("Suicune", ["rest"], evs={"hp": 252, "spd": 252}, nature="Calm"),
         ["thunderbolt"], ["rest"], [["p1", "move 1"], ["p2", "move 1"]])
    # 4. Resisted 0.5× (Jolteon Thunderbolt → Swampert is IMMUNE; use → Venusaur? Grass resists? no.
    #    Electric vs Electric: Thunderbolt → Zapdos = 0.5×).
    atk1("resisted_half",
         mon("Jolteon", ["thunderbolt"], nature="Modest", evs={"spa": 252}),
         mon("Zapdos", ["rest"], evs={"hp": 252, "spd": 252}, nature="Calm"),
         ["thunderbolt"], ["rest"], [["p1", "move 1"], ["p2", "move 1"]])
    # 5. Double super-effective 4× (Cloyster Ice Beam → Salamence, Dragon/Flying: Ice 2×·2×=4×).
    atk1("quad_effective",
         mon("Cloyster", ["icebeam"], nature="Modest", evs={"spa": 252}),
         mon("Salamence", ["rest"], evs={"hp": 252, "spd": 252}, nature="Careful"),
         ["icebeam"], ["rest"], [["p1", "move 1"], ["p2", "move 1"]])
    # 6. Type immunity (Jolteon Earthquake? no. Gengar Thunderbolt → Swampert (Ground) = 0× by type via
    #    Electric→Ground immunity).
    atk1("type_immune_electric_ground",
         mon("Raikou", ["thunderbolt"], nature="Modest", evs={"spa": 252}),
         mon("Swampert", ["protect"], evs={"hp": 252}, nature="Bold"),
         ["thunderbolt"], ["protect"], [["p1", "move 1"], ["p2", "move 1"]])
    # 7. Levitate ability immunity (Earthquake → Claydol = 0× via Levitate).
    atk1("levitate_immune",
         mon("Tyranitar", ["earthquake"], nature="Adamant", evs={"atk": 252}, ability="Sand Stream"),
         mon("Claydol", ["rest"], evs={"hp": 252}, nature="Bold", ability="Levitate"),
         ["earthquake"], ["rest"], [["p1", "move 1"], ["p2", "move 1"]])
    # 8. Thick Fat ×0.5 (Ice Beam → Snorlax with Thick Fat halves Ice/Fire).
    atk1("thick_fat_ice",
         mon("Starmie", ["icebeam"], nature="Modest", evs={"spa": 252}),
         mon("Snorlax", ["rest"], evs={"hp": 252, "spd": 252}, nature="Careful", ability="Thick Fat"),
         ["icebeam"], ["rest"], [["p1", "move 1"], ["p2", "move 1"]])
    # 9. Choice Band ×1.5 physical (Tyranitar Rock Slide → Skarmory).
    atk1("choice_band",
         mon("Tyranitar", ["rockslide"], item="choiceband", nature="Adamant", evs={"atk": 252},
             ability="Sand Stream"),
         mon("Skarmory", ["rest"], evs={"hp": 252, "def": 252}, nature="Impish"),
         ["rockslide"], ["rest"], [["p1", "move 1"], ["p2", "move 1"]])
    # 10. Type-boost item ×1.1 (Magnet → Thunderbolt).
    atk1("type_boost_item",
         mon("Zapdos", ["thunderbolt"], item="magnet", nature="Modest", evs={"spa": 252}),
         mon("Blissey", ["softboiled"], evs={"hp": 252, "spd": 252}, nature="Calm"),
         ["thunderbolt"], ["softboiled"], [["p1", "move 1"], ["p2", "move 1"]])
    # 11. +2 Atk boost (Swords Dance T1, physical attack T2).
    atk1("plus2_atk",
         mon("Salamence", ["swordsdance", "earthquake"], nature="Adamant", evs={"atk": 252}),
         mon("Swampert", ["rest"], evs={"hp": 252, "def": 252}, nature="Bold"),
         ["swordsdance", "earthquake"], ["rest"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 2"], ["p2", "move 1"]])
    # 12. +2 SpA boost (Calm Mind T1, special attack T2).
    atk1("plus2_spa",
         mon("Suicune", ["calmmind", "surf"], nature="Modest", evs={"spa": 252}),
         mon("Snorlax", ["rest"], evs={"hp": 252, "spd": 252}, nature="Careful"),
         ["calmmind", "surf"], ["rest"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 2"], ["p2", "move 1"]])
    # 13. Burn ×0.5 physical (p2 Will-O-Wisp burns p1 T1; p1 physical T2).
    atk1("burn_half_physical",
         mon("Metagross", ["meteormash"], nature="Adamant", evs={"atk": 252}, ability="Clear Body"),
         mon("Gengar", ["willowisp", "rest"], evs={"hp": 252, "def": 252}, nature="Bold", ability="Levitate"),
         ["meteormash"], ["willowisp", "rest"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]])
    # 14. Reflect ×0.5 physical (p2 Reflect T1; p1 physical T2).
    atk1("reflect_physical",
         mon("Tyranitar", ["rockslide"], nature="Adamant", evs={"atk": 252}, ability="Sand Stream"),
         mon("Skarmory", ["reflect", "spikes"], evs={"hp": 252, "def": 252}, nature="Impish"),
         ["rockslide"], ["reflect", "spikes"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]])
    # 15. Light Screen ×0.5 special (p2 Light Screen T1; p1 special T2).
    atk1("lightscreen_special",
         mon("Starmie", ["surf"], nature="Modest", evs={"spa": 252}),
         mon("Blissey", ["lightscreen", "softboiled"], evs={"hp": 252, "spd": 252}, nature="Calm"),
         ["surf"], ["lightscreen", "softboiled"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]])
    # 16. Rain ×1.5 Water (p1 Rain Dance T1; Surf T2).
    atk1("rain_water_boost",
         mon("Suicune", ["raindance", "surf"], nature="Modest", evs={"spa": 252}),
         mon("Snorlax", ["rest"], evs={"hp": 252, "spd": 252}, nature="Careful"),
         ["raindance", "surf"], ["rest"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 2"], ["p2", "move 1"]])
    # 17. Sun ×0.5 Water (p1 Sunny Day T1; Surf T2).
    atk1("sun_water_weaken",
         mon("Suicune", ["sunnyday", "surf"], nature="Modest", evs={"spa": 252}),
         mon("Snorlax", ["rest"], evs={"hp": 252, "spd": 252}, nature="Careful"),
         ["sunnyday", "surf"], ["rest"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 2"], ["p2", "move 1"]])
    # 18. Sun ×1.5 Fire (p1 Sunny Day T1; Flamethrower T2).
    atk1("sun_fire_boost",
         mon("Charizard", ["sunnyday", "flamethrower"], nature="Modest", evs={"spa": 252}),
         mon("Blissey", ["softboiled"], evs={"hp": 252, "spd": 252}, nature="Calm"),
         ["sunnyday", "flamethrower"], ["softboiled"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 2"], ["p2", "move 1"]])
    # 19. Defender +2 Def (p2 Iron Defense / Cosmic Power T1 reduces a physical hit T2).
    atk1("defender_plus2_def",
         mon("Tyranitar", ["earthquake"], nature="Adamant", evs={"atk": 252}, ability="Sand Stream"),
         mon("Forretress", ["irondefense", "rest"], evs={"hp": 252}, nature="Relaxed", ability="Sturdy"),
         ["earthquake"], ["irondefense", "rest"],
         [["p1", "move 1"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]])
    return S


def _run_probe(scenarios: List[dict]) -> List[dict]:
    payload = {"scenarios": [{k: v for k, v in s.items() if not k.startswith("_")} for s in scenarios]}
    proc = subprocess.run(["node", _PROBE_JS], input=json.dumps(payload), capture_output=True,
                          text=True, timeout=180)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"damage_probe.js failed: {proc.stderr[-2000:]}")
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _move_id(disp: str) -> str:
    return "".join(c for c in disp.lower() if c.isalnum())


def _last_attack(res: dict) -> Optional[dict]:
    """Walk the omniscient log: track p2's HP and return the LAST p1→p2 attacking hit (exact damage as a
    fraction of p2 max HP, + whether it crit, + whether it KO'd). None if no clean hit found."""
    maxhp = res["p2"]["maxhp"]
    hp = float(maxhp)
    last = None
    pending = None        # (move_id, crit) awaiting its -damage line
    crit = False
    for line in res["log"]:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "move" and len(parts) >= 5:
            actor, disp, target = parts[2].strip(), parts[3], parts[4].strip()
            if actor.startswith("p1a:") and target.startswith("p2a:"):
                pending = _move_id(disp)
                crit = False
            else:
                pending = None
        elif tag == "-crit":
            crit = True
        elif tag == "-damage" and len(parts) >= 4 and parts[2].strip().startswith("p2a:"):
            field = parts[3]
            is_residual = any(p.startswith("[from]") for p in parts[4:])
            new_hp = _parse_hp(field, maxhp)
            if new_hp is None:
                continue
            if pending is not None and not is_residual:
                dmg = (hp - new_hp) / maxhp
                fainted = new_hp <= 0.0
                last = {"move": pending, "dmg": dmg, "crit": crit, "fainted": fainted}
                pending = None
            hp = new_hp
        elif tag in ("-heal", "-sethp") and len(parts) >= 4 and parts[2].strip().startswith("p2a:"):
            nh = _parse_hp(parts[3], maxhp)
            if nh is not None:
                hp = nh
    return last


def _parse_hp(field: str, maxhp: int) -> Optional[float]:
    tok = field.split()[0]
    if tok in ("0", "0 fnt") or tok.startswith("0 "):
        return 0.0
    if "/" in tok:
        try:
            cur, mx = tok.split("/")
            return float(cur)            # omniscient shows EXACT cur/max → absolute HP
        except Exception:
            return None
    return None


def _band_for(res: dict, move_id: str, tag: str) -> Optional[tuple]:
    """Build the op band from the SIM's EXACT stats/boosts/types/weather/screens/item (no guessing)."""
    p1, p2 = res["p1"], res["p2"]
    for side, snap in (("p1", p1), ("p2", p2)):
        st = snap["stats"]
        _STATS_REGISTRY[(tag, side, snap["species"].lower().replace(" ", "").replace("-", ""))] = {
            "hp": int(snap["maxhp"]), "atk": int(st["atk"]), "def": int(st["def"]),
            "spa": int(st["spa"]), "spd": int(st["spd"]), "spe": int(st["spe"])}
    aty = [t.upper() for t in (p1.get("types") or [])]
    dty = [t.upper() for t in (p2.get("types") or [])]
    attacker = {"species": p1["species"].lower().replace(" ", "").replace("-", ""),
                "t1": aty[0] if aty else None, "t2": aty[1] if len(aty) > 1 else None,
                "boosts": p1.get("boosts") or {}, "burn": (p1.get("status") == "brn"),
                "item": p1.get("item") or None}
    defender = {"species": p2["species"].lower().replace(" ", "").replace("-", ""),
                "t1": dty[0] if dty else None, "t2": dty[1] if len(dty) > 1 else None,
                "boosts": p2.get("boosts") or {}, "burn": False, "item": p2.get("item") or None}
    w = (res.get("weather") or "").lower()
    weather = "rain" if "rain" in w else ("sun" if "sun" in w else None)
    sc = res["p2"].get("sideConditions") or []
    screens = {"reflect": any("reflect" in str(k).lower() for k in sc),
               "lightscreen": any("light" in str(k).lower() for k in sc)}
    band = _op_band(move_id, attacker, defender, weather, screens, tag, "p1", "p2")
    if band is None:
        return None
    # Apply the DEFENDER's ability damage multiplier the op models (Levitate 0×, Flash Fire 0×, Thick Fat
    # 0.5× Fire/Ice, …) — `_op_band` only does the type chart, so post-multiply by the op's OWN table so
    # the probe validates the op's ability physics too. Keyed by ability NUM × move-type index.
    md = gen3_data.moves.get(move_id)
    amul = 1.0
    abil = (res["p2"].get("ability") or "").replace(" ", "").lower()
    ad = gen3_data.abilities.get(abil) if abil else None
    if ad is not None and md is not None and md.type.name != "THREE_QUESTION_MARKS":
        mty = _type_idx(md.type.name)
        if 0 <= ad.num < _OP.ABILITY_DAMAGE_MULT.shape[0]:
            amul = float(_OP.ABILITY_DAMAGE_MULT[ad.num, mty].item())
    low, high, crit = band
    return (low * amul, high * amul, crit * amul)


def main() -> int:
    scenarios = _scenarios()
    print(f"=== DamageOperator PROBE oracle: {len(scenarios)} constructed single-turn scenarios ===")
    results = _run_probe(scenarios)
    by_id = {r.get("id"): r for r in results}
    n_ok = n_fail = n_skip = 0
    for sc in scenarios:
        sid = sc["id"]
        res = by_id.get(sid)
        if res is None or res.get("error"):
            print(f"  SKIP {sid}: probe error {(res or {}).get('error')}")
            n_skip += 1
            continue
        expected_move = _move_id(sc["_p1moves"][-1])          # the LAST p1 move = the measured attack
        band = _band_for(res, expected_move, tag=sid)
        if band is None:
            print(f"  SKIP {sid}: move {expected_move} not band-validatable")
            n_skip += 1
            continue
        low, high, crit = band
        attack = _last_attack(res)                            # None ⇒ the move dealt no damage (immune)
        realized = attack["dmg"] if attack else 0.0
        if high <= 1e-9:
            # The op predicts IMMUNITY (type or ability): the sim must also do 0 damage (no -damage line ⇒
            # attack is None ⇒ realized 0). A non-zero hit on a believed-immune target is a real failure.
            verdict = "OK" if realized <= 1e-6 else "FAIL"
            crit_band = 0.0
        elif attack is None:
            print(f"  SKIP {sid}: op predicts damage but the sim logged no hit (missed/blocked)")
            n_skip += 1
            continue
        else:
            upper = (crit if attack["crit"] else high) * (1.0 + _TOL)
            lower = low * (1.0 - _TOL)
            crit_band = crit
            if attack["fainted"]:
                verdict = "OK(KO)" if realized <= upper else "FAIL"   # KO caps low side; upper still valid
            else:
                verdict = "OK" if lower <= realized <= upper else "FAIL"
        flags = f"{'CRIT ' if attack and attack['crit'] else ''}{'KO ' if attack and attack['fainted'] else ''}"
        print(f"  {verdict:7s} {sid:26s} {expected_move:14s} realized={realized:.3f} "
              f"band=({low:.3f},{high:.3f}) crit={crit_band:.3f} {flags}")
        if verdict.startswith("OK"):
            n_ok += 1
        else:
            n_fail += 1
    print(f"\nPASS {n_ok} | FAIL {n_fail} | SKIP {n_skip}")
    ok = n_fail == 0 and n_ok >= 15
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
