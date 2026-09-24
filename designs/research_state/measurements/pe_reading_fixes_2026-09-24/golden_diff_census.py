"""The GOLDEN obs fixture's diff census for `gen3_pe_reading_fixes_v1` — proven, never regenerated blind.

Two captures of `golden_obs_capture`'s fixed battle set (6 bridge battles, deterministic policy,
fixed sim seed), one with the fork's three reading files at the BASE commit and one with the fix
(`run_census.sh` swaps the files; nothing else differs). Each changed decision is resolved,
index by index, against the DECLARED layout (`Gen3ObservationEncoder.get_layout()` + the
active-context sub-layout from `agents.observation.constants`, never a literal offset) into a
FIELD, and every changed field must be one of the three the fixes touch, VALUE-AWARE against the
decision's own board (recorded beside the vector):

* `context[side].boosts`          — only where that side's active mon is FAINTED (PE-V10), and
                                    the fixed value must be all-zero;
* `context[side].volatile:flashfire` — only where that side's active holds `flashfire` on the
                                    fixed board (PE-V16);
* `<team>[slot].status_counters[toxic]` — only where that slot's mon is badly poisoned (PE-R1b).

Anything else is an UNEXPLAINED change and the census fails. The decision COUNT must also match
(the fixes do not touch the mask, so the deterministic trajectory cannot branch).

    PYTHONPATH=src python golden_diff_census.py capture OUT.npz [N_BATTLES N_TEAMS]
    PYTHONPATH=src python golden_diff_census.py diff BASE.npz FIXED.npz [FIXTURE.json]

With ``N_BATTLES N_TEAMS`` the same deterministic harness plays a WIDER set (the golden's own is
6 × 6) — the census's reach beyond the fixture, so the three fields are each seen at the obs.
"""
from __future__ import annotations

import asyncio
import collections
import json
import sys
from typing import Any, Dict, List

import numpy as np


def capture(out: str, n_battles: str = "", n_teams: str = "") -> None:
    from poke_env import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.battle.gen3_battle import Gen3Battle
    from agents.training import golden_obs_capture as G
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    if n_battles:
        G.N_BATTLES, G.N_TEAMS = int(n_battles), int(n_teams)
    metas: List[Dict[str, Any]] = []

    class _MetaEnc:
        """Wraps the recorder's encoder to note each decision's BOARD beside its vector."""

        def __init__(self, enc):
            self._enc = enc

        def __getattr__(self, k):
            return getattr(self._enc, k)

        def encode(self, battle, **kw):
            live = battle.live_view()
            ours = self._enc.get_team_list(battle, is_opponent=False)
            opp = self._enc.get_team_list(battle, is_opponent=True)

            def slot(mon):
                if mon is None:
                    return None
                return {"species": mon.species, "status": mon.status.name.lower() if mon.status else None,
                        "status_counter": mon.status_counter}

            def act(side):
                a = side.active
                return None if a is None else {"species": a.species, "fainted": a.fainted,
                                               "boosts": dict(a.boosts), "volatiles": sorted(a.volatiles)}

            metas.append({"ours": [slot(m) for m in ours], "opp": [slot(m) for m in opp],
                          "act": [act(live.ours), act(live.opp)]})
            return self._enc.encode(battle, **kw)

    async def _cap():
        from agents.observation import moves as _moves_enc
        _moves_enc._CATEGORY_VAL_CACHE.clear()
        pool = (TeamLoader().get_sample_teams() or TeamLoader().get_all_teams())[:G.N_TEAMS]
        p1 = G._DetPlayer(record=True, battle_format=G.BATTLE_FORMAT, team=G._CyclingTeambuilder(pool),
                          account_configuration=AccountConfiguration("GoldCap", "pw"),
                          server_configuration=LocalhostServerConfiguration, start_listening=False,
                          battle_class=Gen3Battle)
        p1.obs_enc = _MetaEnc(p1.obs_enc)
        p2 = G._DetPlayer(battle_format=G.BATTLE_FORMAT, team=G._CyclingTeambuilder(pool[1:] + pool[:1]),
                          account_configuration=AccountConfiguration("GoldOpp", "pw"),
                          server_configuration=LocalhostServerConfiguration, start_listening=False,
                          battle_class=Gen3Battle)
        await run_local_battles(p1, p2, G.N_BATTLES, seed=G.BRIDGE_SEED)
        return p1.vectors

    vecs = asyncio.run(_cap())
    assert len(vecs) == len(metas), (len(vecs), len(metas))
    hashes = G.vector_hashes(vecs)
    np.savez_compressed(out, vecs=np.stack(vecs), meta=np.array(json.dumps(metas)),
                        hashes=np.array(hashes))
    print(f"captured {len(vecs)} decisions -> {out}")


def _field_namer():
    from agents.observation.constants import BOOSTS_DIM
    from agents.observation.gen3_effects import VOLATILE_SLOTS
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    lay = Gen3ObservationEncoder(mappings=load_mappings()).get_layout()
    parts, mon_lay = lay["parts"], lay["pokemon"]

    def name(idx: int):
        for pname in ("our_team", "opp_team", "context"):
            p = parts[pname]
            if not (p["start"] <= idx < p["end"]):
                continue
            row, col = divmod(idx - p["start"], p["reshape"][1])
            if pname == "context":
                if col < BOOSTS_DIM:
                    return pname, row, "boosts"
                v = col - BOOSTS_DIM
                return pname, row, f"volatile:{VOLATILE_SLOTS[v] if v < len(VOLATILE_SLOTS) else v}"
            best, off_best = "?", -1
            for fname, v in mon_lay.items():
                if isinstance(v, dict) and isinstance(v.get("offset"), int):
                    off, dim = v["offset"], int(v.get("dim", 1))
                    if off <= col < off + dim and off > off_best:
                        best, off_best = f"{fname}[{col - off}]", off
            if best.startswith("status_counters["):
                best = "status_counters[" + ("sleep" if best.endswith("[0]") else "toxic") + "]"
            return pname, row, best
        for pname, p in parts.items():
            if isinstance(p.get("start"), int) and p["start"] <= idx < p["end"]:
                return pname, None, "?"
        return "?", None, "?"

    return name


def diff(base_path: str, fixed_path: str, fixture: str = "") -> int:
    base, fixed = np.load(base_path), np.load(fixed_path)
    bv, fv = base["vecs"], fixed["vecs"]
    fm = json.loads(str(fixed["meta"]))
    print(f"decisions: base {len(bv)}, fixed {len(fv)}; obs dim {bv.shape[1]}")
    if fixture:
        fx = json.load(open(fixture))
        same = sum(a == b for a, b in zip(list(base["hashes"]), fx["hashes"]))
        print(f"BASE capture vs the committed fixture: {same}/{fx['n_decisions']} identical "
              f"(count {len(bv)} vs {fx['n_decisions']})")
    if len(bv) != len(fv):
        print("❌ the decision COUNT changed — the trajectory branched")
        return 1
    name = _field_namer()
    fields = collections.Counter()
    fields_decisions: Dict[str, set] = collections.defaultdict(set)
    indices: Dict[str, set] = collections.defaultdict(set)
    unexplained = []
    changed = 0
    side_key = {"our_team": "ours", "opp_team": "opp"}
    for d in range(len(bv)):
        idx = np.nonzero(bv[d] != fv[d])[0]
        if not len(idx):
            continue
        changed += 1
        for i in idx:
            part, row, fld = name(int(i))
            key = f"{part}[*].{fld}"
            fields[key] += 1
            fields_decisions[key].add(d)
            indices[key].add(int(i))
            meta = fm[d]
            ok = False
            if part == "context" and fld == "boosts":
                a = meta["act"][row]
                ok = bool(a and a["fainted"] and not a["boosts"] and fv[d][i] == 0.0)
                why = "PE-V10"
            elif part == "context" and fld == "volatile:flashfire":
                a = meta["act"][row]
                ok = bool(a and "flashfire" in a["volatiles"] and not a["fainted"])
                why = "PE-V16"
            elif part in side_key and fld == "status_counters[toxic]":
                s = meta[side_key[part]][row]
                ok = bool(s and s["status"] == "tox"
                          and abs(fv[d][i] - min(s["status_counter"], 8) / 8.0) < 1e-6)
                why = "PE-R1b"
            else:
                why = "?"
            if not ok:
                unexplained.append((d, int(i), key, float(bv[d][i]), float(fv[d][i]), why))
    print(f"decisions changed: {changed} / {len(bv)}")
    print(f"{'field (declared layout)':45s} {'decisions':>9s} {'entries':>8s}  indices")
    for key, n in sorted(fields.items()):
        print(f"{key:45s} {len(fields_decisions[key]):9d} {n:8d}  {sorted(indices[key])}")
    detail = [(d, int(i)) for d in range(len(bv)) for i in np.nonzero(bv[d] != fv[d])[0]]
    for d, i in detail[:40]:
        part, row, fld = name(i)
        print(f"   decision {d:4d} idx {i:4d} {part}[{row}].{fld}: base {bv[d][i]:.4f} -> fixed "
              f"{fv[d][i]:.4f}  (board: {fm[d][side_key.get(part, 'ours')][row] if part in side_key else fm[d]['act'][row]})")
    if unexplained:
        print(f"❌ {len(unexplained)} UNEXPLAINED entries (first 20):")
        for u in unexplained[:20]:
            print("   decision %d idx %d %s: base %.6f fixed %.6f (%s)" % u)
        return 1
    print("✅ every changed entry is a field these fixes touch, value-aware against its own board")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(*sys.argv[2:5])
    else:
        sys.exit(diff(*sys.argv[2:5]))
