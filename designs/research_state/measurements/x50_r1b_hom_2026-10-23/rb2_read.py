"""R-b2's REGISTERED READ (``design_x50_rung1b_human_opponent_model.md`` §8.4). Pure numpy over ``rb2_run.py``'s rows
and the bank's ``rows.jsonl``; no model, no core. L3's own read (``l3_read.py``, loaded by path and sha256-checked
through the bank's ``offpool_read.py``) scores each judge's rows cut to L3's shape: its AUC, bootstrap, tie and bars.

* G-REPRO: every state's P* re-score equals the bank's (|Δ| <= 1e-9, the same steps, the same exclusions), and P*'s
  read through L3's code equals the bank's registered READ 2. Otherwise: STOP, no read.
* PRIMARY: the bot stratum's AUC (truth vs the 8 honest worlds) under the HOM, L3's bars verbatim (WORKS lo > 0.555;
  WEAK lo > 0.505; REVERSED hi < 0.495; NO-EVIDENCE hi < 0.545; else NOT-DETECTED).
* CO-PRIMARY: the paired ΔAUC (HOM − P*) per state on the bot stratum, battle-clustered (L3's bootstrap). BETTER iff
  its CI lower > 0, with rule 8's 0.002 band (lower > +0.002).
* VERDICT: HOM-EVIDENCE = WORKS and BETTER; HOM-WEAK = WEAK and BETTER, or WORKS without BETTER; HOM-NONE =
  NO-EVIDENCE or REVERSED; INCONCLUSIVE otherwise.
* Descriptive (never deciding): the snapshot stratum and the pooled AUC under the HOM with ΔAUC; the HOM's AUC_conf
  (a bot-stratum lower > 0.555 attaches FAMILIARITY); top1, p_truth, lifts (L3's blocks); by P*'s outcome; the states
  whose opponent team had a training battle removed by the team-memory exclusion, and the rest.

    python -I rb2_read.py --rows <archive>/rb2_rows.jsonl --train-parts <archive>/train --out rb2_read.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / "search_offpool_bank_2026-10-20"
sys.path.insert(0, str(BANK))
import offpool_common as E  # noqa: E402
import offpool_read as OPR  # noqa: E402  (loads l3_read.py, sha256-checked)

R = OPR.R
DELTA_BAND = 0.002


def better(lo) -> bool:
    return lo is not None and lo > 0.0 + DELTA_BAND


def verdict(auc_v: str, is_better: bool) -> str:
    if auc_v == "WORKS" and is_better:
        return "HOM-EVIDENCE"
    if (auc_v == "WEAK" and is_better) or (auc_v == "WORKS" and not is_better):
        return "HOM-WEAK"
    if auc_v in ("NO-EVIDENCE", "REVERSED"):
        return "HOM-NONE"
    return "INCONCLUSIVE"


def judge_rows(new: List[dict], old: Dict[str, dict], judge: str) -> List[dict]:
    """Each state as an L3-shaped row with ``judge``'s likelihoods (the rest of the row is the bank's)."""
    out = []
    for r in new:
        o = dict(old[r["id"]])
        if r.get("ok"):
            o["ll"] = list(r[f"ll_{judge}"])
            o["conf"] = list(r[f"conf_{judge}"])
            o["n_informative"] = list(r[f"n_informative_{judge}"])
        out.append(OPR.l3_shape(o))
    return out


def removed_teams(train_parts: Path) -> Counter:
    c: Counter = Counter()
    for js in sorted(train_parts.glob("humans_*.json")):
        for k, v in json.loads(js.read_text())["counts"].items():
            if k.startswith("team_memory:"):
                c[k.split(":", 1)[1]] += int(v)
    return c


def read(new: List[dict], old: Dict[str, dict], removed: Counter) -> dict:
    gates: Dict[str, object] = {}
    bad = []
    for r in new:
        o = old[r["id"]]
        if r.get("ok"):
            if not (r.get("repro") or {}).get("ok"):
                bad.append({"id": r["id"], "repro": r.get("repro"), "error": r.get("error")})
        elif o.get("ok") or r.get("defect"):
            bad.append({"id": r["id"], "error": r.get("error"), "old_ok": o.get("ok")})
    gates["n_states"] = len(new)
    gates["G_REPRO_state_failures"] = bad[:20]
    gates["G_REPRO_states"] = len(bad) == 0 and len(new) == len(old)
    P = R.read(judge_rows(new, old, "pstar"))
    H = R.read(judge_rows(new, old, "hom"))
    reg = json.loads((BANK / "result_offpool.json").read_text())["read2_l3_replication"]
    same = all(P[k] == reg[k] for k in ("pooled", "by_stratum"))
    gates["G_REPRO_read2_equal"] = same
    out: Dict[str, object] = {"gates": gates}
    if not (gates["G_REPRO_states"] and same):
        out["stopped"] = True
        out["verdict"] = "STOPPED (G-REPRO)"
        return out
    ok_new = [r for r in new if r.get("ok")]
    ok = [OPR.l3_shape(dict(old[r["id"]])) for r in ok_new]
    bt = R.Boot([r["battle"] for r in ok])
    strata = np.array([r["stratum"] for r in ok])
    Ah = np.array([R.auc_state(r["ll_hom"][0], r["ll_hom"][1:R.J + 1]) for r in ok_new])
    Ap = np.array([R.auc_state(r["ll_pstar"][0], r["ll_pstar"][1:R.J + 1]) for r in ok_new])
    D = Ah - Ap

    def dsum(sel) -> dict:
        e, v, n = bt.mean(D, sel)
        s = R.summ(e, v, n)
        s["BETTER"] = better(s["ci"][0])
        return s

    bot = strata == "bot"
    d_bot = dsum(bot)
    auc_bot = H["by_stratum"]["bot"]["AUC"]
    v = verdict(auc_bot["verdict"], d_bot["BETTER"])
    conf_bot = H["by_stratum"]["bot"]["AUC_conf"]
    outcome = np.array([r["outcome"] for r in ok])
    team = np.array([r["opp_team"] for r in ok])
    rem = np.array([t in removed for t in team])
    desc = {
        "delta_AUC_pooled": dsum(None), "delta_AUC_snapshot": dsum(strata == "snapshot"),
        "familiarity_flag": bool(conf_bot["ci"][0] is not None and conf_bot["ci"][0] > 0.555),
        "bot_by_outcome": {o: {"AUC_hom": R.summ(*bt.mean(Ah, bot & (outcome == o))),
                               "AUC_pstar": R.summ(*bt.mean(Ap, bot & (outcome == o))),
                               "delta": R.summ(*bt.mean(D, bot & (outcome == o)))} for o in ("win", "loss", "draw")},
        "team_memory": {"teams_with_removed_training_battles": len(removed),
                        "bank_states_whose_team_was_removed": int(rem.sum()),
                        "bot_removed": {"AUC_hom": R.summ(*bt.mean(Ah, bot & rem)),
                                        "delta": R.summ(*bt.mean(D, bot & rem))},
                        "bot_not_removed": {"AUC_hom": R.summ(*bt.mean(Ah, bot & ~rem)),
                                            "delta": R.summ(*bt.mean(D, bot & ~rem))}},
    }
    out.update({"hom": H, "pstar": P, "PRIMARY_bot_AUC_hom": auc_bot, "CO_PRIMARY_bot_delta_AUC": d_bot,
                "verdict": v, "descriptive": desc,
                "rule": "HOM-EVIDENCE = WORKS and BETTER; HOM-WEAK = WEAK and BETTER, or WORKS without BETTER; "
                        "HOM-NONE = NO-EVIDENCE or REVERSED; else INCONCLUSIVE (BETTER: ΔAUC CI lower > 0.002)"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--train-parts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    new = R.load(a.rows)
    old = {r["id"]: r for r in R.load(str(E.ARCH / "rows.jsonl"))}
    res = read(new, old, removed_teams(Path(a.train_parts)))
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"verdict": res.get("verdict"), "gates": res["gates"],
                      "primary": res.get("PRIMARY_bot_AUC_hom"), "co_primary": res.get("CO_PRIMARY_bot_delta_AUC")},
                     indent=1, default=str))
    if res.get("stopped"):
        sys.exit(3)


if __name__ == "__main__":
    main()
