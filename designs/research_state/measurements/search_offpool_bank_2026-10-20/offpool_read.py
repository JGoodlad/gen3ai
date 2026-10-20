"""THE REGISTERED READ (``PREREGISTRATION.md`` §7–§9). Pure numpy over the driver's rows (``offpool_run.py``); no model,
no core. It REUSES the two registered reads unchanged, loaded by path and sha256-checked:
  * READ 2, L3 replicated: ``search_l3_2026-10-20/l3_read.py`` ``read()`` on each row cut to L3's shape (the truth + the
    8 honest worlds), i.e. L3's metric, bootstrap, tie band and verdict rule;
  * READ 3, the off-pool decoy: ``counter_test/decoy_read.py`` ``read()`` (its per-state AUCs, bootstrap and rules), with
    its labels mapped to this unit's meaning (§8: a decoy here is an OFF-POOL real team, so its lift is COHERENCE).
Added here, all descriptive: intervals clustered by the OPPONENT TEAM (72 teams recur across battles), and the AUCs
reweighted to L3's bank's outcome mix per stratum (this bank captures every battle, L3's was loss-enriched).

    python -I offpool_read.py --rows <archive>/rows.jsonl --recheck <archive>/recheck.jsonl --out result_offpool.json
    python -I offpool_read.py --selfcheck
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
L3_DIR = HERE.parent / "search_l3_2026-10-20"
#: the registered reads this one reuses (their own registrations: l3_registration.json, decoy_registration.json)
L3_READ_SHA = "036f6260c8cdb39633ccc2a09331ad0519362e2a33b5782a21f46fa9f5489110"
DECOY_READ_FILE = L3_DIR / "counter_test" / "decoy_read.py"
#: L3's bank (its outcome mix per stratum is the reweighting target) and the pool decoy's co-primary, for comparison
L3_STATES = HERE.parent / "hidden_info_maturity_x26_2026-10-18" / "states_mat_x26.json"
POOL_DECOY_DH = {"mean": 0.575, "ci": [0.517, 0.631]}
MIN_STATES = 60


def _load(name: str, path: Path, sha: Optional[str]):
    b = path.read_bytes()
    if sha is not None and hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{path}: sha256 differs from its registration")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


def _decoy_sha() -> str:
    reg = json.loads((L3_DIR / "counter_test" / "decoy_registration.json").read_text())
    return reg["sha256"]["decoy_read.py"]


R = _load("l3_read", L3_DIR / "l3_read.py", L3_READ_SHA)
DR = _load("decoy_read", DECOY_READ_FILE, _decoy_sha())

#: decoy_read's labels -> this unit's (§8): an off-pool decoy's lift over the honest worlds is REAL-TEAM COHERENCE
LABELS = {"POOL-LIFT": "COHERENCE-LIFT", "NO-POOL-LIFT": "NO-LIFT", "TRUTH-PLUS-POOL": "TRUTH-PLUS-COHERENCE",
          "TRUTH-ABOVE-POOL-UNCLEAR": "TRUTH-ABOVE-COHERENCE-UNCLEAR", "FAMILIARITY": "COHERENCE-ONLY"}


def relabel(x):
    if isinstance(x, dict):
        return {("coherence_share_of_L3" if k == "pool_share_of_L3" else k): relabel(v) for k, v in x.items()}
    if isinstance(x, list):
        return [relabel(v) for v in x]
    if isinstance(x, str):
        return LABELS.get(x, x)
    return x


def l3_shape(r: dict) -> dict:
    """A driver row cut to L3's shape: the truth and the J honest worlds only (L3's rows carry no decoy)."""
    o = dict(r)
    if r.get("ok"):
        n = R.J + 1
        o["ll"], o["conf"] = list(r["ll"][:n]), list(r["conf"][:n])
        o["n_informative"] = list(r["n_informative"][:n])
        for k in ("match_m", "exact_m", "hit_u"):
            o[k] = list(r[k][:R.J])
    return o


def replication_label(res: dict) -> str:
    """PREREGISTRATION §7: L3's verdict off-pool, and the A4 labels."""
    v = res.get("verdict")
    if v not in ("WORKS", "WEAK"):
        return {"NO-EVIDENCE": "NOT-REPLICATED", "REVERSED": "NOT-REPLICATED"}.get(v, "INCONCLUSIVE")
    bot = res["by_stratum"].get("bot", {}).get("AUC", {}).get("verdict")
    snap = res["by_stratum"].get("snapshot", {}).get("AUC", {}).get("verdict")
    if bot not in ("WORKS", "WEAK") and snap in ("WORKS", "WEAK"):
        return "REPLICATED-SELF-PLAY-ONLY"
    return "REPLICATED"


def coherence_label(dh_lohi) -> str:
    """PREREGISTRATION §8 (b), READ 3's PRIMARY, on AUC(off-pool decoy vs honest)'s interval."""
    return LABELS[DR.dh_verdict(dh_lohi)] if DR.dh_verdict(dh_lohi) in LABELS else DR.dh_verdict(dh_lohi)


def cluster_auc(x: np.ndarray, clusters: List[str], sel=None) -> dict:
    b = R.Boot(clusters)
    return R.summ(*b.mean(x, sel))


def outcome_weights(rows: List[dict], target: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Per-state weights that give each stratum L3's outcome shares (states equal within a (stratum, outcome) cell)."""
    w = np.zeros(len(rows))
    for s, shares in target.items():
        for o, p in shares.items():
            idx = [i for i, r in enumerate(rows) if r["stratum"] == s and r["outcome"] == o]
            if idx:
                w[idx] = p / len(idx)
    return w


def l3_outcome_mix() -> Dict[str, Dict[str, float]]:
    st = json.loads(L3_STATES.read_text())["states"]
    lean = [v for v in st.values() if v.get("lean")]
    out: Dict[str, Dict[str, float]] = {}
    for s in sorted({v["stratum"] for v in lean}):
        xs = [v["outcome"] for v in lean if v["stratum"] == s]
        out[s] = {o: xs.count(o) / len(xs) for o in sorted(set(xs))}
    return out


def read(rows: List[dict], recheck: Optional[List[dict]], ids_sha256: Optional[str]) -> dict:
    l3 = R.read([l3_shape(r) for r in rows], [l3_shape(r) for r in recheck] if recheck is not None else None,
                ids_sha256)
    out: Dict[str, object] = {"read2_l3_replication": l3}
    if l3.get("stopped"):
        out["stopped"] = True
        return out
    l3["REPLICATION"] = replication_label(l3)
    dec = DR.read(rows, recheck, None, ids_sha256=ids_sha256)
    dec = relabel(dec)
    dec["PRIMARY_coherence"] = coherence_label(dec["pooled"]["AUC_decoy_vs_honest"]["ci"]) if "pooled" in dec else None
    dec["bot_stratum_coherence"] = (coherence_label(dec["by_stratum"]["bot"]["AUC_decoy_vs_honest"]["ci"])
                                    if "bot" in dec.get("by_stratum", {}) else None)
    dec["pool_decoy_dh_225M_pool_bank"] = POOL_DECOY_DH
    if "pooled" in dec:
        lo, hi = dec["pooled"]["AUC_decoy_vs_honest"]["ci"]
        dec["ci_holds_pool_decoy_point"] = lo is not None and lo <= POOL_DECOY_DH["mean"] <= hi
    out["read3_offpool_decoy"] = dec
    out["stopped"] = bool(dec.get("stopped"))

    ok = [r for r in rows if r.get("ok")]
    A = np.array([R.auc_state(r["ll"][0], r["ll"][1:R.J + 1]) for r in ok])
    teams = [r["opp_team"] for r in ok]
    strata = np.array([r["stratum"] for r in ok])
    desc: Dict[str, object] = {
        "team_clustered_AUC_truth_vs_honest": {"pooled": cluster_auc(A, teams),
                                               **{s: cluster_auc(A, teams, strata == s) for s in sorted(set(strata))}},
        "n_opp_teams": len(set(teams))}
    dd = [r for r in ok if r.get("n_decoys", 0) >= 1]
    if dd:
        Adh = np.array([np.mean([R.auc_state(x, r["ll"][1:R.J + 1]) for x in r["ll"][R.J + 1:]]) for r in dd])
        Atd = np.array([np.mean([R.auc_state(r["ll"][0], [x]) for x in r["ll"][R.J + 1:]]) for r in dd])
        tdd = [r["opp_team"] for r in dd]
        desc["team_clustered_AUC_decoy_vs_honest"] = cluster_auc(Adh, tdd)
        desc["team_clustered_AUC_truth_vs_decoy"] = cluster_auc(Atd, tdd)
    mix = l3_outcome_mix()
    w = outcome_weights(ok, mix)
    bt = R.Boot([r["battle"] for r in ok])
    desc["outcome_mix"] = {"this_bank": {s: {o: round(sum(1 for r in ok if r["stratum"] == s and r["outcome"] == o)
                                                      / max(1, int((strata == s).sum())), 4)
                                             for o in ("loss", "win", "draw")} for s in sorted(set(strata))},
                           "l3_bank_lean": {s: {o: round(p, 4) for o, p in v.items()} for s, v in mix.items()}}
    if w.sum() > 0:
        e, v, n = bt.mean(A * w * len(ok), None)
        e2, v2, _ = bt.mean(w * len(ok), None)
        with np.errstate(invalid="ignore", divide="ignore"):
            desc["outcome_reweighted_AUC_truth_vs_honest"] = R.summ(e / e2, v / v2, n)
    desc["by_outcome_AUC_truth_vs_honest"] = {o: R.summ(*bt.mean(A, np.array([r["outcome"] == o for r in ok])))
                                              for o in ("loss", "win", "draw")}
    out["descriptive"] = desc
    out["verdict"] = {"read2": l3["REPLICATION"], "read2_auc_verdict": l3.get("verdict"),
                      "read2_bot": l3["by_stratum"].get("bot", {}).get("AUC", {}).get("verdict"),
                      "read3_primary": dec.get("PRIMARY_coherence"), "read3_combined": dec.get("verdict"),
                      "read3_bot": dec.get("bot_stratum_coherence")}
    return out


def selfcheck() -> None:
    """Planted cases: the cut to L3's shape, the replication labels, the coherence labels, the outcome weights."""
    def fake(shift_t: float, shift_d: float, n: int = 600, seed: int = 3) -> List[dict]:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            ws = rng.normal(0.0, 1.0, R.J).tolist()
            ds = rng.normal(shift_d, 1.0, 2).tolist()
            rows.append({"id": f"s{i}", "battle": f"b{i // 3}", "stratum": "bot" if i % 2 else "snapshot", "ok": True,
                         "ll": [float(rng.normal(shift_t, 1.0))] + ws + ds, "conf": [0.0] * (R.J + 3),
                         "match_m": [0.5] * (R.J + 2), "exact_m": [0.0] * (R.J + 2), "hit_u": [None] * (R.J + 2),
                         "n_informative": [5] * (R.J + 3), "n_scored": 5, "n_unobserved": 0, "n_pending": 0,
                         "n_decoys": 2, "decoy": {"eligible": 3, "shared_sets": [0, 1]}, "opp_team": f"t{i % 40}",
                         "outcome": ("loss", "win", "draw")[i % 3]})
        return rows

    s = read(fake(1.5, 1.5, seed=11), None, None)
    assert s["verdict"]["read2"] == "REPLICATED", s["verdict"]
    assert s["verdict"]["read3_primary"] == "COHERENCE-LIFT", s["verdict"]
    assert s["verdict"]["read3_combined"] in ("NOT-DETECTED", "COHERENCE-ONLY", "TRUTH-PLUS-COHERENCE"), s["verdict"]
    n = read(fake(0.0, 0.0, seed=12), None, None)
    assert n["verdict"]["read2"] in ("NOT-REPLICATED", "INCONCLUSIVE"), n["verdict"]
    assert n["verdict"]["read3_primary"] in ("NO-LIFT", "INCONCLUSIVE"), n["verdict"]
    t = read(fake(1.5, 0.0, seed=13), None, None)
    assert t["verdict"]["read3_primary"] in ("NO-LIFT", "INCONCLUSIVE") and t["verdict"]["read2"] == "REPLICATED"
    assert t["verdict"]["read3_combined"] in ("TRUTH-SPECIFIC", "TRUTH-ABOVE-COHERENCE-UNCLEAR"), t["verdict"]
    # the cut keeps L3's shape exactly
    r0 = fake(0.0, 0.0, 3)[0]
    c = l3_shape(r0)
    assert len(c["ll"]) == R.J + 1 and len(c["match_m"]) == R.J and c["ll"][0] == r0["ll"][0]
    # SELF-PLAY-ONLY: replicated pooled, bot not, snapshot yes
    fake_res = {"verdict": "WEAK", "by_stratum": {"bot": {"AUC": {"verdict": "NOT-DETECTED"}},
                                                  "snapshot": {"AUC": {"verdict": "WORKS"}}}}
    assert replication_label(fake_res) == "REPLICATED-SELF-PLAY-ONLY"
    assert replication_label({"verdict": "NOT-DETECTED", "by_stratum": {}}) == "INCONCLUSIVE"
    assert coherence_label([0.512, 0.6]) == "COHERENCE-LIFT" and coherence_label([0.47, 0.54]) == "NO-LIFT"
    assert coherence_label([0.503, 0.6]) == "INCONCLUSIVE"            # rule 8: inside the band is not clear
    w = outcome_weights([{"stratum": "bot", "outcome": "loss"}, {"stratum": "bot", "outcome": "win"},
                         {"stratum": "bot", "outcome": "win"}], {"bot": {"loss": 0.5, "win": 0.5}})
    assert np.allclose(w, [0.5, 0.25, 0.25])
    print("[offpool read] selfcheck PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows")
    ap.add_argument("--recheck")
    ap.add_argument("--ids-sha256")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    rows = R.load(a.rows)
    rec = R.load(a.recheck) if a.recheck else None
    res = read(rows, rec, a.ids_sha256)
    text = json.dumps(res, indent=1, sort_keys=True, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps(res.get("verdict"), indent=1))
    if res.get("stopped"):
        sys.exit(3)


if __name__ == "__main__":
    main()
