"""Shared helpers of the GIGO audits of the honest-world sampler (audit_*.py). Runs ONLY under run_at_pin.sh.

The pilot folder (value_disc_pilot_2026-10-11) is READ-ONLY and imported via sys.path, put FIRST so that its
``common`` wins over any same-named module in this folder. No outcome is read here: these are validity checks.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
if str(PILOT) in sys.path:
    sys.path.remove(str(PILOT))
sys.path.insert(0, str(PILOT))
import common as C  # noqa: E402  (the pilot's)
import determinize as D  # noqa: E402

assert Path(C.__file__).resolve().parent == PILOT, C.__file__
ARCH = Path.home() / "gen3ai_archive" / "hidden_info_binding"
SRC = Path.home() / "gen3ai_archive" / "value_disc_pilot"
#: the world directories the audits read (the pilot archive's): default the v1 builder's ``det``, ``det_b``; the v2
#: (fixed) builder's re-audit sets ``HIB_AUDIT_DIRS=det2,det2_b,...`` (Smogon first: A5 / A6 / teeth read the first)
#: and ``HIB_AUDIT_TAG=_v2``, which suffixes every output file and the bulk audit directory so the v1 files stand
AUDIT_DIRS = tuple(x for x in os.environ.get("HIB_AUDIT_DIRS", "det,det_b").split(",") if x)
TAG = os.environ.get("HIB_AUDIT_TAG", "")


def builder_ns(plant_off_by_one: bool = False) -> dict:
    """A namespace whose ``build_worlds`` is the pilot's, ALSO returning its command template ``tmpl`` (an
    in-process re-compile; the pilot file is untouched). ``plant_off_by_one`` renders every rewritten move slot one
    position off (the TEETH plant). Patch ``ns["sample_world"]`` to steer the sampler."""
    import inspect
    src = inspect.getsource(D.build_worlds)
    ret = '    return {"teams": teams,'
    assert src.count(ret) == 1, "build_worlds return drifted"
    src = src.replace(ret, '    return {"tmpl": [[s_, list(t) if isinstance(t, tuple) else t] for s_, t in tmpl], "teams": teams,')
    if plant_off_by_one:
        assert src.count("[t[2]]") == 1, "render line drifted"
        src = src.replace("[t[2]]", "[(t[2] + 1) % 4]")
    ns = dict(vars(D))
    exec(compile(src, "<build_worlds_tmpl%s>" % ("_off_by_one" if plant_off_by_one else ""), "exec"), ns)
    return ns


def text_invariant(w: dict, original_cmds) -> dict:
    """A1's TEXT-LEVEL invariant, with the TRUE team as the world: every rendered token equals the ORIGINAL command
    token up to ``toid`` (a rewritten ("MOVE", mon, slot) -> ``move <id>``, ("SLOT", k) -> ``switch <name>``; every
    other token byte-equal), and the template's rewritten-token counts equal the builder's ``leak`` counts."""
    tmpl, cm = w["tmpl"], w["cmds"][0]
    orig = [list(x) for x in original_cmds]
    if cm is None or len(cm) != len(orig) or len(tmpl) != len(orig):
        return {"ok": False, "reason": "no world / length differs", "mismatches": []}
    bad = []
    n_move = n_slot = 0
    for i, ((_s_t, t), (s_r, r), (s_o, o)) in enumerate(zip(tmpl, cm, orig)):
        if isinstance(t, list):
            n_move += t[0] == "MOVE"
            n_slot += t[0] == "SLOT"
            vr, _, ar = r.partition(" ")
            vo, _, ao = o.partition(" ")
            if not (s_r == s_o and vr == vo and D.toid(ar) == D.toid(ao)):
                bad.append({"i": i, "template": t, "rendered": r, "original": o})
        elif [s_r, r] != [s_o, o]:
            bad.append({"i": i, "template": t, "rendered": r, "original": o})
    counts_ok = n_move == w["leak"]["rewritten_moves"] and n_slot == w["leak"]["rewritten_switches"]
    return {"ok": not bad and counts_ok, "rewritten_moves": n_move, "rewritten_switches": n_slot,
            "leak_counts": [w["leak"]["rewritten_moves"], w["leak"]["rewritten_switches"]], "counts_match": counts_ok,
            "n_mismatch": len(bad), "mismatches": bad[:5]}


def cp() -> str:
    c = C.cp()
    C.check_pin(c)
    return c


def audit_dir(c: str) -> Path:
    d = ARCH / c / ("audit" + TAG)
    d.mkdir(parents=True, exist_ok=True)
    return d


def bank(c: str):
    from main.policy_spectrum.bank import load_bank
    return load_bank(SRC / c / "bank")


def priors():
    return D.Priors(Path(os.environ["VDP_PIN"]) / "data" / "pokemon")


def phase1_rows(c: str) -> Dict[str, dict]:
    return {r["id"]: r for q in sorted((SRC / c / "rows").glob("shard_*.jsonl")) for r in C.read_rows(q) if r.get("ok")}


def world_rows(c: str, det_dir: str) -> Dict[str, dict]:
    return {r["id"]: r for q in sorted((SRC / c / det_dir).glob("worlds_*.jsonl")) for r in C.read_rows(q)}


def played_ids(c: str, det_dir: str) -> set:
    """States whose det playout row is ok (the played worlds = the first 8 of each). A directory with NO playout rows
    yet (a re-audit before its playouts) falls back to every state with 8 built worlds (the worlds that would play)."""
    rows = sorted((SRC / c / det_dir).glob("rows_*.jsonl"))
    if not rows:
        return {i for i, w in world_rows(c, det_dir).items()
                if w.get("ok") and len(w["teams"]) >= 8 and all(t is not None for t in w["teams"][:8])}
    return {r["id"] for q in rows for r in C.read_rows(q) if r.get("ok")}


def decision(bk, did: str):
    d = next(x for x in bk.decisions if x["id"] == did)
    b = bk.battles[bk.battle_index[d["battle"]]]
    return d, b


def rec(b, label: str, opp_key: str, team_opp, commands):
    from agents.battle.core_replay import RecordedBattle
    p = {"p1": dict(b.p1), "p2": dict(b.p2)}
    if team_opp is not None:
        p[opp_key] = dict(p[opp_key], team=team_opp)
    return RecordedBattle(label=label, format_id=b.format_id, seed=b.seed, p1=p["p1"], p2=p["p2"],
                          commands=[list(x) for x in commands])


def obs_entries(res: dict, viewer: int) -> List[dict]:
    return [e for e in res["trackers"][viewer] if "obs" in e]


def row_of(e: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(e["obs"]["b64"]), dtype="<f4")


def write_json(name: str, obj) -> Path:
    if TAG and name.endswith(".json") and not name[:-5].endswith(TAG):
        name = name[:-5] + TAG + ".json"
    p = HERE / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n")
    return p


# ------------------------------------------------------------------------------------------------
# cell names through the pin's observation layout
# ------------------------------------------------------------------------------------------------

def cell_names() -> List[dict]:
    """One dict per obs index: {block, slot (team-row index or None), field, sub}. Built from the pin's
    ``Gen3ObservationEncoder.get_layout()`` and ``agents.observation.constants``."""
    from agents.observation import constants as K
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    L = Gen3ObservationEncoder(load_mappings()).get_layout()
    total = int(L["total_dim"])
    out: List[dict] = [None] * total  # type: ignore[list-item]
    full = int(K.POKEMON_FULL_DIM)
    pl = L["pokemon"]
    mon = [None] * full
    for name, spec in pl.items():
        if not isinstance(spec, dict):
            continue
        o, n = int(spec["offset"]), int(spec["dim"])
        sub = spec.get("layout")
        for k in range(n):
            f, s = name, str(k) if n > 1 else ""
            if name == "moves":
                f, s = f"move{k // int(K.MOVE_SLOT_DIM)}", [x for x, v in sub["slot_layout"].items()
                                                           if int(v["offset"]) <= k % int(K.MOVE_SLOT_DIM) < int(v["offset"]) + int(v["dim"])][0]
            elif isinstance(sub, dict):
                for sn, sv in sub.items():
                    if isinstance(sv, dict) and int(sv["offset"]) <= k < int(sv["offset"]) + int(sv["dim"]):
                        s = sn + (f"[{k - int(sv['offset'])}]" if int(sv["dim"]) > 1 else "")
            mon[o + k] = (f, s)
    extra = {int(K.POKEMON_SLEEP_BELIEF_OFFSET): ("sleep_belief", int(K.POKEMON_SLEEP_BELIEF_DIM)),
             int(K.POKEMON_RECENCY_OFFSET): ("recency", int(K.POKEMON_RECENCY_DIM)),
             int(K.POKEMON_LAST_ACTION_OFFSET): ("last_action", int(K.POKEMON_LAST_ACTION_DIM)),
             int(K.POKEMON_TRAPPED_OFFSET): ("trapped", 1), int(K.POKEMON_MAYBE_TRAPPED_OFFSET): ("maybe_trapped", 1),
             int(K.POKEMON_ACTIVE_OFFSET): ("active", 1)}
    for o, (nm, n) in extra.items():
        for k in range(n):
            if mon[o + k] is None:
                mon[o + k] = (nm, str(k) if n > 1 else "")
    for k in range(full):
        if mon[k] is None:
            mon[k] = ("mon_unnamed", str(k))
    for team, base in (("own_team", int(K.OFFSET_OUR_TEAM)), ("other_team", int(K.OFFSET_OPP_TEAM))):
        for s in range(int(K.TEAM_SIZE)):
            for k in range(full):
                out[base + s * full + k] = {"block": team, "slot": s, "field": mon[k][0], "sub": mon[k][1], "k": k}
    ctx = int(K.ACTIVE_CONTEXT_DIM)
    for side, nm in enumerate(("own_active_ctx", "other_active_ctx")):
        for k in range(ctx):
            f = "boosts" if k < int(K.BOOSTS_DIM) else "volatiles"
            out[int(K.OFFSET_CONTEXT) + side * ctx + k] = {"block": nm, "slot": None, "field": f,
                                                           "sub": str(k if f == "boosts" else k - int(K.BOOSTS_DIM))}
    for nm, spec in L["global_layout"].items():
        for k in range(int(spec["dim"])):
            out[int(K.OFFSET_GLOBAL) + int(spec["offset"]) + k] = {"block": "global", "slot": None, "field": nm, "sub": str(k)}
    for nm, spec in L["reactive_layout"].items():
        for k in range(int(spec["dim"])):
            out[int(K.OFFSET_REACTIVE) + int(spec["offset"]) + k] = {"block": "reactive", "slot": None, "field": nm, "sub": str(k)}
    T, cd = int(K.TEAM_SIZE), int(K.PAIR_HISTORY_CELL_DIM)
    for k in range(int(K.PAIR_HISTORY_DIM)):
        i, j, c_ = k // (T * cd), (k // cd) % T, k % cd
        out[int(K.OFFSET_PAIR_HISTORY) + k] = {"block": "pair_history", "slot": i, "field": f"h[i,{j}]", "sub": str(c_)}
    for k in range(int(K.EVENT_WINDOW_DIM)):
        out[int(K.OFFSET_EVENT_WINDOW) + k] = {"block": "event_window", "slot": None,
                                               "field": f"row{k // int(K.EVENT_TOKEN_DIM)}", "sub": str(k % int(K.EVENT_TOKEN_DIM))}
    for nm, spec in L["obs_facts"].items():
        for k in range(int(spec["dim"])):
            out[int(K.OFFSET_OBS_FACTS) + int(spec["offset"]) + k] = {"block": "obs_facts", "slot": None, "field": nm, "sub": str(k)}
    miss = [i for i, x in enumerate(out) if x is None]
    if miss:
        raise RuntimeError(f"unnamed obs cells: {miss[:10]} (+{len(miss)})")
    return out
