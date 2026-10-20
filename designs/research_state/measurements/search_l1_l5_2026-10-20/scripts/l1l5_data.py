"""L1 / L5 DATA: one compact per-state cache over the three BANKED honest-search banks (`../PREREG.md` §2).
PURE numpy over the archives (no model, no engine, no playout). READ-ONLY on every input: it never calls a bank
module's directory helpers (several `mkdir` their archive paths); every path is built here.

Banks (each state: the policy's root logits over the legal actions, the referee T(a) = the mean outcome over the TRUE
world's referee seeds 16..39, and per SOURCE the per-action, per-world arrays W (outcome), X4 (the critic at the
searched side's first decision >= T + 4, else the terminal) and crit (the one-ply critic), all in POLICY order):

  tier1  the hidden-info study's Tier 1 (end2 + prod at 15M; registered 600 + supplement per checkpoint), arm F (and
         FN), the inclusion and the cross-check from the FROZEN blinded table (sha256 committed beside it)
  e2     the competent-opponent bank E2 (end2 + prod; 600 states each), arm F, the inclusion by `read_e2.records`'
         rules (truth ok; F's world ok; F's row ok), the cross-check against `result_e2.json`'s pooled IMP per source
  x26    the maturity battery's INTERIM bank, LEAN ids, anchors 25M / 100M / 175M / 225M, arms F (and FN), the
         registered read's COMMON states (truth ok, F ok, no root tie at every anchor; `read_mat.common_ids`), the
         cross-check per state against the frozen LEAN table (`score/table.pkl`, sha256 beside it)

    python -I l1l5_data.py build      → <cache>/cache.pkl (+ cache.json: counts, input digests, the cross-checks)

The cache stores the TRUTH's arrays too (O: leaf seeds 0..7, the descriptive oracle rows; and the 40 outcomes and end
turns for L5).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parents[1]
T1_DIR = MEAS / "hidden_info_tier1_2026-10-13"
E2_DIR = MEAS / "hidden_info_e2_competent_2026-10-16"
MAT_DIR = MEAS / "hidden_info_maturity_x26_2026-10-18"
PILOT_DIR = MEAS / "value_disc_pilot_2026-10-11"
HOME = Path.home()
T1_ARCH = HOME / "gen3ai_archive" / "hidden_info_tier1"
PILOT_ARCH = HOME / "gen3ai_archive" / "value_disc_pilot"
E2_ARCH = HOME / "gen3ai_archive" / "hidden_info_e2"
MAT_ARCH = HOME / "gen3ai_archive" / "hidden_info_maturity"
CACHE = Path(os.environ.get("L1L5_CACHE") or (HOME / "gen3ai_archive" / "search_l1_l5"))

sys.path.insert(0, str(T1_DIR))
import t1_data as D  # noqa: E402  (pure: numpy + t1_common's constants)

CPS = ("end2", "prod")
X26_ANCHORS = ("x26_025M", "x26_100M", "x26_175M", "x26_225M")
J = 8
SCORERS = ("LEAF-K8", "TRUNC-4-K8", "CRITIC")


# ------------------------------------------------------------------------------------------------
# read-only row readers
# ------------------------------------------------------------------------------------------------

def read_rows(p: Path) -> List[dict]:
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:            # a torn last line (the study's own reader ignores it too)
                pass
    return out


def read_dir(d: Path, pattern: str) -> Dict[str, dict]:
    """{id: the LAST row with that id} (the study readers' rule: a re-run row supersedes)."""
    out: Dict[str, dict] = {}
    if d.exists():
        for p in sorted(d.glob(pattern)):
            for r in read_rows(p):
                out[r["id"]] = r
    return out


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------
# per-state arrays
# ------------------------------------------------------------------------------------------------

def legal_logits(logits, acts: List[str]) -> np.ndarray:
    return np.array([float(logits[int(a)]) for a in acts], dtype=np.float64)


def truth_part(r1: dict, logits, top) -> dict:
    """The truth (Phase 1) in POLICY order: Tp, the legal logits, O's K8 arrays, the 40 outcomes and end turns."""
    A = D.truth_arrays(r1, {"logits": logits, "top": top})
    o = np.array(A["order"])
    acts = A["acts"]
    Ob = np.array([[D.DEC[ch] for ch in r1["out"][a]] for a in acts])[o]                  # [A, 40] in {-1, 0, +1}
    et = r1.get("end_turn")
    end = np.array([et[a] for a in acts], dtype=np.int64)[o] if et else None              # [A, 40]
    return {"acts": acts, "order": A["order"], "turn0": A["turn"],
            "rec": {"logit": legal_logits(logits, acts)[o], "Tp": A["T"][o],
                    "O": {"W": A["W"][o][:, :J], "X4": A["Xd"][4][o][:, :J], "crit": A["crit"][o]},
                    "Ob40": Ob.astype(np.int8), "end40": end, "turn0": A["turn"]}}


def world_part(rd: dict, tp: dict) -> dict:
    Aw = D.world_arrays(rd, tp["acts"], tp["turn0"])
    o = np.array(tp["order"])
    return {"W": Aw["W"][o], "X4": Aw["Xd"][4][o], "crit": Aw["crit"][o]}


def first_max(s: np.ndarray) -> int:
    m = s.max()
    return int(np.argmax(s >= m - D.TIE))


def argmax_imps(src: dict, Tp: np.ndarray) -> Dict[str, float]:
    """The registered acting rule (the first action in the policy's order within TIE of the max) per scorer."""
    out = {}
    for z, key in (("LEAF-K8", "W"), ("TRUNC-4-K8", "X4"), ("CRITIC", "crit")):
        out[z] = float(Tp[first_max(np.asarray(src[key], dtype=np.float64).mean(1))] - Tp[0])
    return out


# ------------------------------------------------------------------------------------------------
# TIER 1
# ------------------------------------------------------------------------------------------------

def load_tier1(log: dict) -> List[dict]:
    tb = T1_ARCH / "blinded_table.pkl"
    want = (T1_DIR / "blinded_table.sha256").read_text().split()[0]
    got = sha_file(tb)
    if got != want:
        raise SystemExit(f"tier1: blinded table sha256 {got} != committed {want}")
    key = json.loads((T1_DIR / "arm_key.json").read_text())["codes"]
    code = {v["label"]: c for c, v in key.items()}
    tab = pickle.loads(tb.read_bytes())
    log["tier1"] = {"blinded_table_sha256": got, "codes": {"F": code["F"], "FN": code["FN"]}, "per_cp": {}}
    out = []
    for cp in CPS:
        reg = json.loads((PILOT_DIR / f"states_{cp}.json").read_text())["states"]
        supp = json.loads((T1_DIR / f"states_supp_{cp}.json").read_text())["states"]
        r1 = {**read_dir(PILOT_ARCH / cp / "rows", "shard_*.jsonl"), **read_dir(T1_ARCH / cp / "rows_supp", "shard_*.jsonl")}
        rows = {lab: read_dir(T1_ARCH / cp / "rows" / code[lab], "rows_*.jsonl") for lab in ("F", "FN")}
        n = {"table": 0, "F_ok": 0, "FN_ok": 0, "argmax_mismatch": 0}
        for t in tab[cp]["recs"]:
            n["table"] += 1
            if not (t["src"].get("O", {}).get("ok") and t["src"].get(code["F"], {}).get("ok")):
                continue
            e = reg.get(t["id"]) or supp[t["id"]]
            tp = truth_part(r1[t["id"]], e["logits"], e["top"])
            rec = {"bank": "tier1", "cp": cp, "id": t["id"], "battle": t["battle"], "stratum": cp, **tp["rec"],
                   "k": t["k"], "turn": t["turn"], "material": t["material"], "opp": t["opp_class"]}
            rec["F"] = world_part(rows["F"][t["id"]], tp)
            if not np.allclose(rec["Tp"], t["Tp"], atol=1e-12):
                raise SystemExit(f"tier1 {t['id']}: Tp differs from the blinded table")
            mine = argmax_imps(rec["F"], rec["Tp"])
            reg_imp = {z: t["src"][code["F"]]["imp"][z] for z in SCORERS}
            if any(abs(mine[z] - reg_imp[z]) > 1e-9 for z in SCORERS):
                n["argmax_mismatch"] += 1
            rec["reg_imp_F"] = reg_imp
            n["F_ok"] += 1
            if t["src"].get(code["FN"], {}).get("ok") and t["id"] in rows["FN"]:
                rec["FN"] = world_part(rows["FN"][t["id"]], tp)
                n["FN_ok"] += 1
            out.append(rec)
        log["tier1"]["per_cp"][cp] = n
    return out


# ------------------------------------------------------------------------------------------------
# E2
# ------------------------------------------------------------------------------------------------

def load_e2(log: dict) -> List[dict]:
    res = json.loads((E2_DIR / "result_e2.json").read_text())
    log["e2"] = {"result_e2_sha256": sha_file(E2_DIR / "result_e2.json"), "per_cp": {}}
    out = []
    for cp in CPS:
        st = json.loads((E2_DIR / f"states_e2_{cp}.json").read_text())["states"]
        truth = read_dir(E2_ARCH / cp / "rows" / "O", "shard_*.jsonl")
        rows = read_dir(E2_ARCH / cp / "rows" / "F", "rows_*.jsonl")
        worlds = read_dir(E2_ARCH / cp / "worlds" / "F", "worlds_*.jsonl")
        n = {"states": len(st), "truth_missing": 0, "truth_refused": 0, "F_excluded_or_defect": 0, "F_missing": 0,
             "F_refused": 0, "F_ok": 0}
        for did in sorted(st):
            e = st[did]
            r1 = truth.get(did)
            if r1 is None:
                n["truth_missing"] += 1
                continue
            if not r1.get("ok"):
                n["truth_refused"] += 1
                continue
            w = worlds.get(did)
            if w is not None and not w.get("ok"):
                n["F_excluded_or_defect"] += 1
                continue
            rd = rows.get(did)
            if rd is None:
                n["F_missing"] += 1
                continue
            if not rd.get("ok"):
                n["F_refused"] += 1
                continue
            tp = truth_part(r1, e["logits"], e["top"])
            rec = {"bank": "e2", "cp": cp, "id": did, "battle": f"{cp}:{e['battle']}", "stratum": cp, **tp["rec"],
                   "k": int(e["k"]), "turn": int(e["turn"]), "material": int(e["material"]), "opp": "snapshot"}
            rec["F"] = world_part(rd, tp)
            n["F_ok"] += 1
            out.append(rec)
        log["e2"]["per_cp"][cp] = n
    # the cross-check: pooled mean and n of IMP(F) per scorer against the registered read
    chk = {}
    for z in SCORERS:
        reg = res["scorers"][z]["pooled"]["IMP per source"]["F"]
        mine = [argmax_imps(r["F"], r["Tp"])[z] for r in out]
        chk[z] = {"registered": reg, "mine_mean": float(np.mean(mine)), "mine_n": len(mine)}
    log["e2"]["crosscheck"] = chk
    return out


# ------------------------------------------------------------------------------------------------
# X26 (the maturity battery's interim LEAN rows)
# ------------------------------------------------------------------------------------------------

def load_x26(log: dict) -> List[dict]:
    tp_ = MAT_ARCH / "score" / "table.pkl"
    want = (MAT_ARCH / "score" / "table.sha256").read_text().split()[0]
    got = sha_file(tp_)
    if got != want:
        raise SystemExit(f"x26: LEAN table sha256 {got} != its sha256 file {want}")
    tab = pickle.loads(tp_.read_bytes())
    st = json.loads((MAT_DIR / "states_mat_x26.json").read_text())["states"]
    lean = sorted(x for x, e in st.items() if str(e["lean"]) == "True" or e["lean"] is True)
    by = {lab: {r["id"]: r for r in tab[lab]["recs"]} for lab in X26_ANCHORS}
    common = []
    for did in lean:
        rs = [by[lab].get(did) for lab in X26_ANCHORS]
        if any(r is None for r in rs) or any(r["tie"] for r in rs):
            continue
        if all(r["src"].get("O", {}).get("ok") and r["src"].get("F", {}).get("ok") for r in rs):
            common.append(did)
    reg = json.loads((MAT_DIR / "result_mat.json").read_text())
    csha = hashlib.sha256("".join(common).encode()).hexdigest()
    if csha != reg["common_ids_sha256"] or got != reg["table_sha256"] or len(common) != reg["n_common"]:
        raise SystemExit("x26: the common states / the LEAN table differ from the registered read's (result_mat.json)")
    log["x26"] = {"lean_table_sha256": got, "lean_ids": len(lean), "common": len(common), "common_ids_sha256": csha,
                  "per_anchor": {}}
    out = []
    for lab in X26_ANCHORS:
        roots = read_dir(MAT_ARCH / "roots", f"{lab}.jsonl")
        truth = read_dir(MAT_ARCH / "rows" / lab / "O", "shard_*.jsonl")
        rows = {arm: read_dir(MAT_ARCH / "rows" / lab / arm, "rows_*.jsonl") for arm in ("F", "FN")}
        n = {"F_ok": 0, "FN_ok": 0, "argmax_mismatch": 0}
        for did in common:
            rt = roots[did]
            tp = truth_part(truth[did], rt["logits"], rt["top"])
            m = st[did]
            rec = {"bank": "x26", "cp": lab, "id": did, "battle": m["battle"], "stratum": "x26", **tp["rec"],
                   "k": int(m["k"]), "turn": int(m["turn"]), "material": int(m["material"]), "opp": m["stratum"]}
            rec["F"] = world_part(rows["F"][did], tp)
            reg_imp = {z: by[lab][did]["src"]["F"]["imp"][z] for z in SCORERS}
            mine = argmax_imps(rec["F"], rec["Tp"])
            if any(abs(mine[z] - reg_imp[z]) > 1e-9 for z in SCORERS):
                n["argmax_mismatch"] += 1
            rec["reg_imp_F"] = reg_imp
            n["F_ok"] += 1
            fr = rows["FN"].get(did)
            if fr is not None and fr.get("ok") and by[lab][did]["src"].get("FN", {}).get("ok"):
                rec["FN"] = world_part(fr, tp)
                n["FN_ok"] += 1
            out.append(rec)
        log["x26"]["per_anchor"][lab] = n
    return out


# ------------------------------------------------------------------------------------------------

def x26_battle_reveals(log: dict) -> List[dict]:
    """L5's disambiguation input: per X26 interim-bank battle, from the TRACED side's own protocol view (the copied
    eval traces), the opponent facts revealed turn by turn (species, moves, items, abilities) and the true team's
    fact counts. Read-only."""
    out = []
    n_missing = 0
    with gzip.open(MAT_ARCH / "bank" / "battles.jsonl.gz", "rt") as f:
        battles = [json.loads(x) for x in f if x.strip()]
    for b in battles:
        bid, side = b["battle_id"], b["traced_side"]
        run, cyc, opp, name = bid.split("/")
        p = MAT_ARCH / "traces" / run / "eval_traces" / cyc / opp / f"{name}.{side}.jsonl.gz"
        if not p.exists():
            n_missing += 1
            continue
        opp_side = "p2" if side == "p1" else "p1"
        team = parse_team(b[opp_side]["team"])
        events = []                      # (turn, kind, actor, value)
        with gzip.open(p, "rt") as f:
            next(f)                      # the header record
            for line in f:
                r = json.loads(line)
                for rd in r.get("readings") or []:
                    if rd.get("side") != "opp":
                        continue
                    k = rd.get("kind")
                    v = rd.get("value") or {}
                    if k in ("SWITCH", "DRAG"):
                        events.append((rd["turn"], "species", rd.get("actor"), None))
                    elif k == "MOVE":
                        if "[from]" in r["text"]:                 # a called move (Sleep Talk, Metronome, ...) is no slot
                            continue
                        events.append((rd["turn"], "move", rd.get("actor"), v.get("move_id")))
                    elif k == "ABILITY":
                        events.append((rd["turn"], "ability", rd.get("actor"), v.get("ability")))
                    elif k in ("ITEM", "ENDITEM"):
                        events.append((rd["turn"], "item", rd.get("actor"), v.get("item")))
                    if "[from] item:" in r["text"] and "[of]" not in r["text"] and k != "ITEM":
                        events.append((rd["turn"], "item", rd.get("actor"), None))     # e.g. a Leftovers heal
        out.append({"battle": bid, "side": side, "team": team, "events": events, "outcome": b["outcome"],
                    "stratum": "bot" if opp in BOT_OPPS else "snapshot"})
    log["x26_reveals"] = {"battles": len(battles), "traces_missing": n_missing, "parsed": len(out)}
    return out


BOT_OPPS = {"random", "heuristic", "heuristic2", "staller", "staller_v2", "aggressive", "aggressive_v2", "setup_sweep",
            "setup_sweep_v2"}


def parse_team(packed: str) -> List[dict]:
    """Showdown's packed team: name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness]..."""
    mons = []
    for s in packed.split("]"):
        f = s.split("|")
        if len(f) < 5:
            continue
        species = (f[1] or f[0]).lower()
        mons.append({"species": "".join(ch for ch in species if ch.isalnum()),
                     "moves": [m for m in f[4].split(",") if m], "item": f[2], "ability": f[3]})
    return mons


def build() -> None:
    log: dict = {"status": "the L1 / L5 cache (inputs only; no rule evaluated)"}
    recs = load_tier1(log) + load_e2(log) + load_x26(log)
    rev = x26_battle_reveals(log)
    CACHE.mkdir(parents=True, exist_ok=True)
    b = pickle.dumps({"recs": recs, "reveals": rev}, protocol=4)
    (CACHE / "cache.pkl").write_bytes(b)
    log["cache_sha256"] = hashlib.sha256(b).hexdigest()
    log["n"] = {bk: sum(r["bank"] == bk for r in recs) for bk in ("tier1", "e2", "x26")}
    (CACHE / "cache.json").write_text(json.dumps(log, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(log, indent=1, sort_keys=True, default=str))


def load_cache() -> dict:
    return pickle.loads((CACHE / "cache.pkl").read_bytes())


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        raise SystemExit("usage: l1l5_data.py build")
    build()
