"""TIER 1's READ, part 1: the per-state arrays every score is computed from. PURE numpy over the archives (no model, no
engine). Read-only on every input.

Sources per state (each on the pilot's win scale [0, 1]; the referee T(a) = the mean outcome over the TRUE world's
truth seeds 16..39, which no scorer reads):

  O        the truth (Phase 1: the pilot's registered rows, this unit's supplement rows `rows_supp`): leaf seeds 0..15,
           the value trace, the one-ply critic on seeds 40..47
  P        the pilot's Phase 2 base re-built on the FIXED builder (v2, `det2`; Amendment 1 (b)); registered states only
  Tnn      one coded Tier-1 arm (`rows/<code>/`): 8 worlds, world j on dice seed j; its honest one-ply critic world j on
           seed 40 + j
  H…       the hand-evaluator re-plays (`hand_eval_playouts.py`, codes `hand_eval_arm_key.json`: HE@O / HE@P / HE@F):
           truncated at T + 5, each searched-side decision's critic V and Foul Play's hand eval h, and the one-ply
           successors' V and h

Scorers (per action; IMP = T(the scorer's argmax, ties toward the policy's order) − T(the policy's top1)):
  LEAF-K{1,4,8}, TRUNC-{1,2,4}-K{4,8} (the critic V at the searched side's first decision ≥ T + d, else the terminal),
  CRITIC (the mean of the 8 one-ply successors); for HE sources HE-TRUNC-{1,2,4}-K{4,8} and HE-1PLY on Foul Play's
  own leaf scale σ(0.0125·(h − h_root)) with a terminal 1 / ½ / 0 (README §6: one scale for every HE scorer, because a
  K-mean mixes terminals with evaluations), and VT-TRUNC-4-K8 / VT-CRITIC from the SAME re-plays' V (the paired critic
  side of R-EVAL; identical to the arm's own TRUNC-4-K8 / CRITIC by construction, checked).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402

TIE = 1e-6
DEC = {"+": 1.0, "-": -1.0, "0": 0.0}
TRUNC_D = (1, 2, 4)
SCORERS = ["LEAF-K1", "LEAF-K4", "LEAF-K8"] + [f"TRUNC-{d}-K{k}" for d in TRUNC_D for k in (4, 8)] + ["CRITIC"]
HE_SCORERS = [f"HE-TRUNC-{d}-K{k}" for d in TRUNC_D for k in (4, 8)] + ["HE-1PLY", "VT-TRUNC-4-K8", "VT-CRITIC"]
HE_CODES = {"H8e895870": "HE@O", "H8503aeeb": "HE@P", "H528ede2b": "HE@F"}     # public (committed key)
HE_LEAF_SCALE = 0.0125


def w(x):
    return (np.asarray(x, dtype=np.float64) + 1.0) / 2.0


def pick(score: np.ndarray, order: List[int]) -> int:
    """The pilot's rule: the first action, in the POLICY's order, within TIE of the maximum."""
    m = max(score[i] for i in order)
    return next(i for i in order if score[i] >= m - TIE)


def fp_leaf(h: float, h_root: float) -> float:
    return 1.0 / (1.0 + math.exp(-HE_LEAF_SCALE * (h - h_root)))


# ------------------------------------------------------------------------------------------------
# loaders
# ------------------------------------------------------------------------------------------------

def phase1_rows(cp: str, states: str) -> Dict[str, dict]:
    d = (T.PILOT_ARCH / cp / "rows") if states == "registered" else (T.cp_dir(cp) / "rows_supp")
    return T.read_dir(d, "shard_*.jsonl")


def p_rows(cp: str) -> Dict[str, dict]:
    return T.read_dir(T.PILOT_ARCH / cp / "det2", "rows_*.jsonl")


def code_rows(cp: str, code: str) -> Dict[str, dict]:
    return T.read_dir(T.cp_dir(cp) / "rows" / code, "rows_*.jsonl")


def code_worlds(cp: str, code: str) -> Dict[str, dict]:
    return T.read_dir(T.cp_dir(cp) / "worlds" / code, "worlds_*.jsonl")


def he_rows(cp: str, code: str) -> Dict[str, dict]:
    return T.read_dir(T.cp_dir(cp) / "hand_eval" / code, "rows_*.jsonl")


def coded_dirs(cp: str) -> List[str]:
    d = T.cp_dir(cp) / "rows"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("T")) if d.exists() else []


def stage_meta(cp: str) -> Dict[str, dict]:
    """{id: {set, battle, k, turn, material, bins, outcome, cycle_opp, logits, top}} over the registered 600 and the
    supplement."""
    out: Dict[str, dict] = {}
    reg = T.registered_states(cp)
    sf = json.loads((T.BRC / f"stage_facts_{cp}.json").read_text())["states"]
    for did, e in reg.items():
        k = 6 - int(e["unrevealed"])
        mat = int(sf[did]["alive_ours"]) + int(sf[did]["alive_theirs"])
        out[did] = {"set": "registered", "k": k, "turn": int(e["turn"]), "material": mat, **_common(cp, e)}
    for did, e in T.supplement_states(cp).items():
        if did in out:
            raise RuntimeError(f"{did}: in the registered set and the supplement")
        out[did] = {"set": "supp", "k": int(e["k"]), "turn": int(e["turn"]), "material": int(e["material"]),
                    **_common(cp, e)}
    for did, m in out.items():
        m.update({"k_bin": T.k_bin(m["k"]), "turn_bin": T.turn_bin(m["turn"]), "mat_bin": T.mat_bin(m["material"])})
    return out


def _common(cp: str, e: dict) -> dict:
    return {"battle": f"{cp}:{e['battle']}", "outcome": e["outcome"], "cycle_opp": e["cycle_opp"],
            "logits": e["logits"], "top": e["top"], "opp_class": e["cycle_opp"].split("/")[1]}


# ------------------------------------------------------------------------------------------------
# per-state arrays
# ------------------------------------------------------------------------------------------------

def truth_arrays(r: dict, meta: dict) -> dict:
    """Phase 1 (the TRUE world): acts, the policy's order, top1, T, and the oracle components."""
    acts = sorted(r["out"], key=int)
    Ob = np.array([[DEC[ch] for ch in r["out"][a]] for a in acts])           # [A, 40]
    T0 = int(r["turn"])
    lg = np.array([meta["logits"][int(a)] for a in acts], dtype=np.float64)
    order = sorted(range(len(acts)), key=lambda i: (-lg[i], int(acts[i])))
    if int(acts[order[0]]) != int(meta["top"][0]):
        raise RuntimeError(f"{r['id']}: top1 {acts[order[0]]} != registered {meta['top'][0]}")
    Xd = {d: np.empty((len(acts), T.N_LEAF)) for d in TRUNC_D}
    for i, a in enumerate(acts):
        for s in range(T.N_LEAF):
            t = r["trace"][a][s]
            term = w(Ob[i, s])
            for d in TRUNC_D:
                k = next((q for q, x in enumerate(t["t"]) if x >= T0 + d), None)
                Xd[d][i, s] = t["v"][k] if k is not None else term
    crit = w(np.array([r["crit"][a]["q"] for a in acts], dtype=np.float64))   # [A, 8] seeds 40..47
    return {"acts": acts, "order": order, "top1": order[0], "turn": T0,
            "T": w(Ob[:, T.N_LEAF:T.N_LEAF + T.N_TRUTH]).mean(1),
            "W": w(Ob[:, :T.N_LEAF]), "Xd": Xd, "crit": crit}


def world_arrays(rd: dict, acts: List[str], T0: int) -> dict:
    """A det-format row (a coded arm, or P): W [A, 8], Xd [A, 8], crit [A, 8] per world."""
    if sorted(rd["out"], key=int) != acts:
        raise RuntimeError(f"{rd['id']}: the arm's actions differ from Phase 1's")
    Ob = np.array([[DEC[ch] for ch in rd["out"][a]] for a in acts])
    if Ob.shape[1] != T.J:
        raise RuntimeError(f"{rd['id']}: {Ob.shape[1]} worlds, not {T.J}")
    Xd = {d: np.empty((len(acts), T.J)) for d in TRUNC_D}
    for i, a in enumerate(acts):
        for j in range(T.J):
            t = rd["trace"][a][j]
            term = w(Ob[i, j])
            for d in TRUNC_D:
                k = next((q for q, x in enumerate(t["t"]) if x >= T0 + d), None)
                Xd[d][i, j] = t["v"][k] if k is not None else term
    crit = w(np.array([rd["crit"][a]["q"] for a in acts], dtype=np.float64))
    return {"W": w(Ob), "Xd": Xd, "crit": crit}


def scorer_values(A: dict) -> Dict[str, np.ndarray]:
    """{scorer: per-action score} for truth / world arrays (K counts seeds or worlds 0..K-1)."""
    out = {f"LEAF-K{K}": A["W"][:, :K].mean(1) for K in (1, 4, 8)}
    for d in TRUNC_D:
        for K in (4, 8):
            out[f"TRUNC-{d}-K{K}"] = A["Xd"][d][:, :K].mean(1)
    out["CRITIC"] = A["crit"].mean(1)
    return out


END_W = {"+": 1.0, "-": 0.0, "0": 0.5}


def he_arrays(rh: dict, acts: List[str], T0: int, tally: dict) -> Dict[str, np.ndarray]:
    """A hand-eval row → {HE scorer: per-action score} (README §6)."""
    if sorted(rh["trace"], key=int) != acts:
        raise RuntimeError(f"{rh['id']}: hand-eval actions differ from Phase 1's")
    n = int(rh.get("n", 8))
    h_root = float(rh["h_root"])
    H = {d: np.empty((len(acts), n)) for d in TRUNC_D}
    V4 = np.empty((len(acts), n))
    for i, a in enumerate(acts):
        for j in range(n):
            t = rh["trace"][a][j]
            e = rh["end"][a][j]
            for d in TRUNC_D:
                k = next((q for q, x in enumerate(t["t"]) if x >= T0 + d), None)
                if k is not None:
                    H[d][i, j] = fp_leaf(float(t["h"][k]), h_root)
                    if d == 4:
                        V4[i, j] = float(t["v"][k])
                elif e[0] in END_W:
                    H[d][i, j] = END_W[e[0]]
                    if d == 4:
                        V4[i, j] = END_W[e[0]]
                else:                                   # truncated with no decision at >= T + d (counted)
                    tally["truncated_without_decision"] = tally.get("truncated_without_decision", 0) + 1
                    H[d][i, j] = fp_leaf(float(t["h"][-1]), h_root) if t["h"] else 0.5
                    if d == 4:
                        V4[i, j] = float(t["v"][-1]) if t["v"] else 0.5
    h1 = np.empty((len(acts), n))
    v1 = np.empty((len(acts), n))
    for i, a in enumerate(acts):
        op = rh["one_ply"][a]
        for j in range(n):
            src = op["src"][j]
            if src in END_W:
                h1[i, j] = v1[i, j] = END_W[src]
            else:
                h1[i, j] = fp_leaf(float(op["h"][j]), h_root)
                v1[i, j] = float(op["v"][j])
    out = {}
    for d in TRUNC_D:
        for K in (4, 8):
            out[f"HE-TRUNC-{d}-K{K}"] = H[d][:, :K].mean(1)
    out["HE-1PLY"] = h1.mean(1)
    out["VT-TRUNC-4-K8"] = V4[:, :8].mean(1)
    out["VT-CRITIC"] = v1.mean(1)
    return out
