"""gen3_beatup_exact_v1 — Beat Up against the REAL gen-3 sim, on CONSTRUCTED scenarios (the omniscient
`utils/bridge/damage_probe.js` BattleStream: exact HP, the sim's own `[of]` ally list, a fixed PRNG seed per
run). Three layers, each deterministic (fixed seeds):

  1. MECHANICS — the sim's own log names the allies each hit is struck for (``-activate|…|move: Beat Up|[of] X``).
     The set is EXACTLY the healthy party (fainted / statused allies absent, the user present iff healthy);
     the hit count equals it; and EVERY realised hit lies inside the integer band the gen-3 formula allows for
     that ally's base Atk vs the target's base Def — `floor((floor(floor(420*A/D)/50)*screen + 2) * r / 100)`,
     r in 85..100 — so typeless, base-stat-only, per-hit-+2 and the screen rule are all pinned to the engine.
  2. THE SIM'S MEAN — the mean over the seeds of the total damage sits on the exact integer mean of that
     process (the per-roll floors included), within 3 standard errors.
  3. THE OP — `DamageOperator`'s smooth 0.925-mean price for the same configuration is within the rounding the
     smooth formula is documented to carry (<= 1.5 HP per hit) of that integer mean, in HP.

Run as a script (`python src/agents/model/beatup_sim_parity_test.py [--json out.json]`) to print / save the table
`designs/research_state/measurements/beatup_golden_2026-10-03/` records.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Dict, List, Optional

import pytest
import torch

from agents import gen3_data
from agents.model.damage_op_test import _fake_ctx_out, _make_layout
from agents.model.features_extractor import DamageOperator, TEAM_SIZE
from agents.observation.constants import POKEMON_CONDITION_OFFSET
from agents.observation.types import TypeEncoder
from utils.paths import src_path

# The omniscient constructed-scenario probe (`utils/bridge/damage_probe.js`) and its three helpers, lifted
# from the deleted `poke_env_gaps/damage_op_probe_fuzz_test.py` (T27 P6 slice 6d-2) — this test was their
# last user.
_PROBE_JS = str(src_path("utils", "bridge", "damage_probe.js"))
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

_T2I = TypeEncoder.TYPE_TO_IDX
_BEATUP = gen3_data.moves.get("beatup").num
_SEEDS = [[s, 2, 3, 4] for s in range(1, 25)]                      # fixed: the same battles every run
_PARTY = ["umbreon", "tyranitar", "salamence", "snorlax", "metagross", "gengar"]
# Seeds on which the paralysed Umbreon is NOT fully paralysed on turn 2 (found once by scanning 1..60, then PINNED —
# a precondition the scenario asserts, never a branch). Seeds 1 and 2 are full-para turns: no Beat Up to measure.
_NOT_FULLY_PARALYSED = (3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17)


def _base(species: str, stat: str) -> int:
    return int(gen3_data.species.get(species).base_stats[stat])


def _atk_set(party: List[str]) -> Dict[str, int]:
    return {p: _base(p, "atk") for p in party}


def _sets(p1_names: List[str], p2_name: str, p2_moves: List[str], p2_evs: Optional[dict] = None) -> tuple:
    p1 = [mon(n, ["beatup", "splash", "calmmind"] if n == "umbreon" else ["splash"], evs={"hp": 0})
          for n in p1_names]
    p2 = [mon(p2_name, p2_moves, evs=p2_evs or {"hp": 252, "spd": 252}, nature="Calm")]
    return p1, p2


def _scenario(name: str, party: List[str], target: str, target_moves: List[str], choices, *,
              expect_allies: List[str], screen: Optional[str] = None, target_evs: Optional[dict] = None,
              seeds=None) -> dict:
    p1, p2 = _sets(party, target, target_moves, target_evs)
    return {"id": name, "formatid": "gen3customgame", "p1": p1, "p2": p2, "choices": choices,
            "_party": list(party), "_target": target, "_allies": expect_allies, "_screen": screen,
            "_seeds": seeds or _SEEDS}


def _scenarios() -> List[dict]:
    S: List[dict] = []
    beat = [["p1", "move 1"], ["p2", "move 1"]]
    S.append(_scenario("six_healthy_into_blissey", _PARTY, "blissey", ["softboiled"], beat,
                       expect_allies=list(_PARTY)))
    # typeless into a Ghost, a Steel, a Dark and a Psychic (the dex type, Dark, would be 2x / 0.5x / 0.5x / 2x)
    for tgt in ("gengar", "skarmory", "tyranitar", "alakazam"):
        S.append(_scenario(f"typeless_into_{tgt}", _PARTY, tgt, ["splash"], beat, expect_allies=list(_PARTY)))
    # Light Screen halves, Reflect does not (p2 sets it T1; p1 Umbreon Splashes; T2 Beat Up)
    S.append(_scenario("light_screen_halves", _PARTY, "blissey", ["lightscreen", "softboiled"],
                       [["p1", "move 2"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]],
                       expect_allies=list(_PARTY), screen="light"))
    S.append(_scenario("reflect_does_not", _PARTY, "blissey", ["reflect", "softboiled"],
                       [["p1", "move 2"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]],
                       expect_allies=list(_PARTY), screen="reflect"))
    # the user's +2 SpA and the target's +2 Def change nothing
    S.append(_scenario("stages_do_not_reach_it", _PARTY, "blissey", ["irondefense", "softboiled"],
                       [["p1", "move 3"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]],
                       expect_allies=list(_PARTY)))
    # a PARALYSED ally and a SLEEPING ally are excluded: they take the status while active, then bench
    st = ["snorlax", "tyranitar", "umbreon", "salamence", "metagross", "gengar"]
    S.append(_scenario("statused_allies_excluded", st, "blissey", ["thunderwave", "spore", "softboiled"],
                       [["p1", "move 1"], ["p2", "move 1"],           # T1 Snorlax Splashes, Thunder Wave -> par
                        ["p1", "switch 2"], ["p2", "move 2"],         # T2 Tyranitar in, Spore -> slp
                        ["p1", "switch 3"], ["p2", "move 3"],         # T3 Umbreon in
                        ["p1", "move 1"], ["p2", "move 3"]],          # T4 Beat Up
                       expect_allies=["umbreon", "salamence", "metagross", "gengar"]))
    # a FAINTED ally is excluded: Shedinja (1 HP) is KO'd by Shadow Ball, Umbreon is sent in
    fnt = ["shedinja", "umbreon", "tyranitar", "salamence", "snorlax", "metagross"]
    p1 = [mon("shedinja", ["splash"], evs={"hp": 0}, ability="Wonder Guard")] + [
        mon(n, ["beatup", "splash"] if n == "umbreon" else ["splash"], evs={"hp": 0}) for n in fnt[1:]]
    S.append({"id": "fainted_ally_excluded", "formatid": "gen3customgame", "p1": p1,
              "p2": [mon("blissey", ["shadowball", "softboiled"], evs={"hp": 252, "spd": 252}, nature="Calm")],
              "choices": [["p1", "move 1"], ["p2", "move 1"], ["p1", "switch 2"],
                          ["p1", "move 1"], ["p2", "move 2"]],
              "_party": fnt, "_target": "blissey", "_allies": fnt[1:], "_screen": None, "_seeds": _SEEDS})
    # a PARALYSED USER does not count itself (Thunder Wave T1, Beat Up T2): seeds where it is not fully paralysed
    S.append(_scenario("statused_user_excluded", _PARTY, "blissey", ["thunderwave", "softboiled"],
                       [["p1", "move 2"], ["p2", "move 1"], ["p1", "move 1"], ["p2", "move 2"]],
                       expect_allies=_PARTY[1:], seeds=[[s, 2, 3, 4] for s in _NOT_FULLY_PARALYSED]))
    return S


def _last_beatup(log: List[str], target_name: str) -> Optional[dict]:
    """The LAST Beat Up use in an omniscient log: the `[of]` allies in order, each hit's exact HP loss, crits."""
    start = None
    for i, line in enumerate(log):
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "move" and parts[3] == "Beat Up":
            start = i
    if start is None:
        return None
    allies, hits, crits = [], [], 0
    for line in log[start + 1:]:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "move" or tag == "turn":
            break
        if tag == "-activate" and len(parts) >= 5 and parts[3] == "move: Beat Up":
            allies.append(parts[4].replace("[of] ", "").strip())
        elif tag == "-crit":
            crits += 1
        elif tag == "-damage" and parts[2].startswith("p2a:") and len(parts) >= 4 and not parts[3:][-1].startswith("[from]"):
            new = _parse_hp(parts[3], 0)
            if new is None:
                continue
            hits.append(new)
    # hit i's damage = the HP before it - after it; the first 'before' is the run's p2 maxhp-after-prior-turns,
    # taken from the last `|-damage|`/`switch` HP seen BEFORE the move
    before = None
    for line in reversed(log[:start]):
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] in ("-damage", "-heal", "switch") and parts[2].startswith("p2a:"):
            before = _parse_hp(parts[3].split()[0], 0) if parts[1] != "switch" else _parse_hp(parts[4], 0)
            if before is not None:
                break
    if before is None:
        return None
    dmg, prev = [], before
    for h in hits:
        dmg.append(prev - h)
        prev = h
    return {"allies": allies, "dmg": dmg, "crits": crits}


def _band(ally_atk: int, target_def: int, screen: bool):
    """The integer damage range of ONE non-crit hit: floor((floor(floor(420*A/D)/50)*scr + 2) * r / 100)."""
    b = math.floor(math.floor(420 * ally_atk / target_def) / 50)
    if screen:
        b = math.floor(b * 0.5)
    top = b + 2
    return math.floor(top * 85 / 100), top


def _integer_mean(allies: List[str], target_def: int, screen: bool) -> float:
    tot = 0.0
    for a in allies:
        b = math.floor(math.floor(420 * _base(a, "atk") / target_def) / 50)
        if screen:
            b = math.floor(b * 0.5)
        tot += sum(math.floor((b + 2) * r / 100) for r in range(85, 101)) / 16.0
    return tot


def _op_hp(op, party: List[str], target: str, expect_allies: List[str], screen: Optional[str]) -> float:
    """The op's smooth mean price (HP) of OUR Beat Up for this configuration, read from `_outgoing_block`."""
    ctx = _fake_ctx_out(our_species=gen3_data.species.get(party[0]).num, our_t1=_T2I["DARK"], our_t2=0,
                        our_moves=[_BEATUP, 0, 0, 0], our_move_types=[_T2I["DARK"], 0, 0, 0],
                        opp_species=gen3_data.species.get(target).num, opp_t1=_T2I["NORMAL"], opp_t2=0,
                        move_mask=[1, 0, 0, 0])
    # the user (umbreon) in slot 0; an excluded ally is FAINTED to the op if the sim fainted it, else STATUSED
    order = ["umbreon"] + [p for p in party if p != "umbreon"]
    for i, sp in enumerate(order):
        ctx.species_ids[:, i] = gen3_data.species.get(sp).num
        ctx.hp_and_active[:, i, 0] = 0.0 if sp == "shedinja" else 1.0
        if sp not in expect_allies and sp != "shedinja":
            ctx.pokemon_part[:, i, POKEMON_CONDITION_OFFSET + 3] = 1.0          # asleep: ineligible, still alive
    ctx.opp_believed_mask = torch.zeros(1, TEAM_SIZE, dtype=torch.bool)
    ctx.screen_feature[:, 3] = 1.0 if screen == "light" else 0.0
    ctx.screen_feature[:, 1] = 1.0 if screen == "reflect" else 0.0
    maxhp = 2.0 * _base(target, "hp") + 31.0 + 110.0
    return op._outgoing_block(ctx)[0, 1].item() * maxhp


def run_all() -> List[dict]:
    scen = _scenarios()
    flat = []
    for sc in scen:
        for sd in sc["_seeds"]:
            flat.append({**{k: v for k, v in sc.items() if not k.startswith("_")}, "seed": sd,
                         "id": f"{sc['id']}#{sd[0]}"})
    results = _run_probe(flat)
    by_scen: Dict[str, List[dict]] = {}
    for r in results:
        assert "error" not in r and "fatal" not in r, r
        by_scen.setdefault(r["id"].split("#")[0], []).append(r)
    op = DamageOperator(_make_layout(), outgoing=True)
    out = []
    for sc in scen:
        target = sc["_target"]
        tdef = _base(target, "def")
        screen = sc["_screen"] == "light"
        runs = []
        for r in by_scen[sc["id"]]:
            bu = _last_beatup(r["log"], target)
            assert bu is not None and bu["dmg"], (sc["id"], r["id"])       # the PRECONDITION: Beat Up was used
            bu["allies"] = [a.lower() for a in bu["allies"]]
            runs.append(bu)
        party = sc["_party"]
        allies = sc["_allies"]
        in_band = all(
            _band(_base(a, "atk"), tdef, screen)[0] <= d <= _band(_base(a, "atk"), tdef, screen)[1]
            for bu in runs if bu["crits"] == 0 for a, d in zip(bu["allies"], bu["dmg"]))
        n_ok = all(sorted(bu["allies"]) == sorted(allies) and len(bu["dmg"]) == len(allies) for bu in runs)
        clean = [bu for bu in runs if bu["crits"] == 0]
        sim_mean = sum(sum(bu["dmg"]) for bu in clean) / len(clean)
        want = _integer_mean(allies, tdef, screen)
        var = sum((_band(_base(a, "atk"), tdef, screen)[1] * 0.15) ** 2 / 12.0 for a in allies)   # uniform 85..100
        se = math.sqrt(var / len(clean))
        op_hp = _op_hp(op, party, target, allies, sc["_screen"])
        out.append({"scenario": sc["id"], "n_runs": len(runs), "n_clean": len(clean), "allies_ok": n_ok,
                    "all_hits_in_band": in_band, "sim_mean_hp": round(sim_mean, 3),
                    "integer_mean_hp": round(want, 3), "se_hp": round(se, 3),
                    "op_smooth_mean_hp": round(op_hp, 3), "op_minus_integer_mean": round(op_hp - want, 3),
                    "hits": len(allies)})
    return out


@pytest.mark.sim
def test_beat_up_matches_the_real_sim():
    rows = run_all()
    for r in rows:
        assert r["allies_ok"], r                                                    # layer 1: the ally set + count
        assert r["all_hits_in_band"], r                                             #          every hit in its band
        assert abs(r["sim_mean_hp"] - r["integer_mean_hp"]) <= 3.0 * r["se_hp"] + 0.5, r   # layer 2
        assert abs(r["op_minus_integer_mean"]) <= 1.5 * r["hits"], r                # layer 3: the op's smooth mean


if __name__ == "__main__":
    rows = run_all()
    hdr = ("scenario", "n_clean", "hits", "sim_mean_hp", "integer_mean_hp", "se_hp", "op_smooth_mean_hp",
           "op_minus_integer_mean", "allies_ok", "all_hits_in_band")
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[h]) for h in hdr))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(rows, f, indent=1)
