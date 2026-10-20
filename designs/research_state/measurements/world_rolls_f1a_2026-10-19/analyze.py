"""F1 tier A: the read of step 1 (``screen_shadow.py``) and step 5 (``rebuild_dice.py``). Descriptive only.

    $PY -I analyze.py --out-dir <dir>      # reads <dir>/step1/shadow_*.jsonl, <dir>/step5/{seed,rolls}.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List


def _rows(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def step1(rows: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for stratum in ("random", "excluded", "all"):
        rs = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
        agg: Dict[str, Counter] = {"screen": Counter(), "joint": Counter()}
        for r in rs:
            for kind in agg:
                agg[kind].update(r["counts"].get(kind, {}))
        sc = agg["screen"]
        out[stratum] = {
            "states": len(rs),
            "today_ok": sum(1 for r in rs if r["ok"]),
            "defects": sum(1 for r in rs if r.get("crash")),
            "screen_replays": sc["n"], "screen_accept_today": sc["today"], "screen_accept_tier_a": sc["tier_a"],
            "screen_rejected_today_accepted_tier_a": sc["rescued"],
            "share_of_today_rejections_tier_a_accepts": round(sc["rescued"] / max(1, sc["n"] - sc["today"]), 4),
            "acceptance_ratio_tier_a_over_today": round(sc["tier_a"] / max(1, sc["today"]), 3),
            "today_accepts_tier_a_rejects (must be 0)": sc["today_only"] + agg["joint"]["today_only"],
            "states_with_a_rescue": sum(1 for r in rs if r["counts"].get("screen", {}).get("rescued", 0) > 0),
            "excluded_today_with_a_rescued_proposal": sum(1 for r in rs if not r["ok"]
                                                          and r["counts"].get("screen", {}).get("rescued", 0) > 0),
            "sites_mean": round(sum(r.get("sites", 0) for r in rs) / max(1, len(rs)), 1),
            "pinned_sites_share": round(sum(r.get("pinned_sites", 0) for r in rs)
                                        / max(1, sum(r.get("sites", 0) for r in rs)), 4),
            "rescued_widened_sites_mean": round(sum(sum(r["widened_sites"]) for r in rs)
                                                / max(1, sum(len(r["widened_sites"]) for r in rs)), 2),
        }
    return out


def _tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def _entries(team: str) -> List[List[str]]:
    return [e.split("|") for e in team.split("]")]


def step5(seed: List[dict], rolls: List[dict]) -> dict:
    S = {r["id"]: r for r in seed}
    R = {r["id"]: r for r in rolls}
    ids = sorted(set(S) & set(R))
    out: dict = {"states_both": len(ids)}
    for stratum in ("random", "excluded", "all"):
        ii = [i for i in ids if stratum == "all" or S[i]["stratum"] == stratum]
        ex_s = sum(1 for i in ii if not S[i]["ok"])
        ex_r = sum(1 for i in ii if not R[i]["ok"])
        out[f"exclusions_{stratum}"] = {
            "states": len(ii), "seed": ex_s, "rolls": ex_r,
            "seed_rate": round(ex_s / max(1, len(ii)), 4), "rolls_rate": round(ex_r / max(1, len(ii)), 4),
            "recovered (seed excluded, rolls ok)": sum(1 for i in ii if not S[i]["ok"] and R[i]["ok"]),
            "lost (seed ok, rolls excluded)": sum(1 for i in ii if S[i]["ok"] and not R[i]["ok"]),
            "defects": {"seed": sum(1 for i in ii if S[i].get("crash")), "rolls": sum(1 for i in ii if R[i].get("crash"))},
        }
    both = [i for i in ids if S[i]["ok"] and R[i]["ok"]]
    out["ok_in_both"] = len(both)
    # roll widening and weights (rolls)
    wid = [w for i in ids if R[i]["ok"] for w in R[i]["widened"]]
    ess = []
    for i in ids:
        if R[i]["ok"] and R[i]["log_w"]:
            lw = R[i]["log_w"]
            m = max(lw)
            w = [math.exp(x - m) for x in lw]
            ess.append(sum(w) ** 2 / sum(x * x for x in w))
    out["rolls_worlds"] = {
        "worlds": len(wid), "share_off_stream": round(sum(1 for w in wid if w) / max(1, len(wid)), 4),
        "off_stream_rolls_mean": round(sum(wid) / max(1, len(wid)), 3),
        "weight_ess_of_8_mean": round(sum(ess) / max(1, len(ess)), 3),
        "weight_ess_of_8_min": round(min(ess), 3) if ess else None,
        "states_ess_below_4": sum(1 for e in ess if e < 4),
    }
    # the marginals, on the states ok in both
    def rate(rows: Dict[str, dict], key: str) -> float:
        n = k = 0
        for i in both:
            for fl in rows[i]["facts"]:
                for f in fl:
                    if key in f:
                        n += 1
                        k += bool(f[key])
        return round(k / max(1, n), 4)

    out["seen_fact_match_with_truth"] = {m: {"seed": rate(S, c), "rolls": rate(R, c)}
                                        for m, c in (("item", "item"), ("ability", "ability"), ("spread", "spread"))}
    out["seen_exact_hp_equal_truth"] = {
        "seed": round(sum(S[i]["seen_exact_hp_equal"] for i in both) / max(1, sum(S[i]["seen_mons"] for i in both)), 4),
        "rolls": round(sum(R[i]["seen_exact_hp_equal"] for i in both) / max(1, sum(R[i]["seen_mons"] for i in both)), 4)}
    out["regate_pass"] = {"seed": sum(1 for i in both if S[i]["regate"]), "rolls": sum(1 for i in both if R[i]["regate"]),
                          "of": len(both)}
    # distinct spreads per seen slot across the 8 worlds (the posterior's width), and per-slot TVs between modes
    def per_slot(row: dict) -> Dict[int, Dict[str, Counter]]:
        tt = _entries(row["true_team"])
        d: Dict[int, Dict[str, Counter]] = {}
        for t in row["teams"]:
            for s, f in enumerate(_entries(t)):
                if f[0] != tt[s][0]:
                    continue
                c = d.setdefault(s, {"item": Counter(), "spread": Counter(), "species": Counter()})
                c["item"][f[2]] += 1
                c["spread"][(f[5], f[6])] += 1
        return d

    dist = {"seed": [], "rolls": []}
    tv = {"item": [], "spread": []}
    sp_tv = []
    for i in both:
        a, b = per_slot(S[i]), per_slot(R[i])
        for s in set(a) & set(b):
            dist["seed"].append(len(a[s]["spread"]))
            dist["rolls"].append(len(b[s]["spread"]))
            for k in tv:
                tv[k].append(_tv(a[s][k], b[s][k]))
        ca = Counter(f[0] or f[1] for t in S[i]["teams"] for f in _entries(t))
        cb = Counter(f[0] or f[1] for t in R[i]["teams"] for f in _entries(t))
        sp_tv.append(_tv(ca, cb))
    out["seen_slot_distinct_spreads_of_8_mean"] = {k: round(sum(v) / max(1, len(v)), 3) for k, v in dist.items()}
    out["seen_slot_tv_seed_vs_rolls_mean"] = {k: round(sum(v) / max(1, len(v)), 4) for k, v in tv.items()}
    out["team_species_tv_seed_vs_rolls_mean"] = round(sum(sp_tv) / max(1, len(sp_tv)), 4)
    out["cost_wall_s_mean"] = {"seed": round(sum(S[i]["wall_s"] for i in ids) / max(1, len(ids)), 1),
                               "rolls": round(sum(R[i]["wall_s"] for i in ids) / max(1, len(ids)), 1)}
    out["screens_mean_ok_both"] = {"seed": round(sum(S[i]["screens"] for i in both) / max(1, len(both)), 1),
                                   "rolls": round(sum(R[i]["screens"] for i in both) / max(1, len(both)), 1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--write", default=None, help="write the read as JSON here")
    a = ap.parse_args()
    d = Path(a.out_dir)
    s1 = [r for p in sorted((d / "step1").glob("shadow_*.jsonl")) for r in _rows(p)]
    res = {"step1": step1(s1), "step5": step5(_rows(d / "step5" / "seed.jsonl"), _rows(d / "step5" / "rolls.jsonl"))}
    # step 1's today-builder verdicts must equal step 5's seed builds (the same code path and keys)
    s5 = {r["id"]: r for r in _rows(d / "step5" / "seed.jsonl")}
    res["step1_vs_step5_seed_ok_disagree"] = sum(1 for r in s1 if r["id"] in s5 and r["ok"] != s5[r["id"]]["ok"])
    txt = json.dumps(res, indent=2, sort_keys=True)
    print(txt)
    if a.write:
        Path(a.write).write_text(txt + "\n")


if __name__ == "__main__":
    main()
