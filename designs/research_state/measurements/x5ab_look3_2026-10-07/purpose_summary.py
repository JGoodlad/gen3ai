"""X5 A/B LOOK 3: the compact record of purpose metric (1) (``purpose_look3.sh``'s output) → ``purpose/summary.json``.

The 16 full read JSONs (~2.7 MB each) are archived outside the repo (their sha256 is recorded here); this keeps, per
run, the ``per_run`` block (on-pool, the values ``infer`` reads), the conditional read's mean block, its exclusion
counts per reference, the eset reference, the reencode gate and the reader commit; plus ``infer``'s JSON.

    python purpose_summary.py [--dir purpose] [--archive DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compact(d: dict) -> dict:
    ic = d["on_pool"]["intent_conditional"]
    refs = []
    for r in ic["per_reference"]:
        a = r["all"]
        refs.append({"reference_label": r["reference_label"], "logloss": a["logloss"],
                     "logloss_ci95_battle_clustered": a["logloss_ci95"], "n_rows": a["n_rows"],
                     "n_scored": a["n_scored"], "n_zero_event_mass": a["n_zero_event_mass"],
                     "set_miss_rate": a["set_miss_rate"], "excluded_rule8_tie": r["excluded_rule8_tie"],
                     "excluded_rule8_denominator": r["excluded_rule8_denominator"],
                     "excluded_struggle": r["excluded_struggle"],
                     "coverage": {k: r["coverage"][k] for k in ("mean_outside_mass", "outside_freq", "citl",
                                                                "citl_ci95", "z", "n")},
                     "by_kind_logloss": {k: r[k]["logloss"] for k in ("move", "switch")}})
    return {"label": d["label"], "arm": d["arm"], "checkpoint": d["checkpoint"], "per_run": d["per_run"],
            "intent_conditional_mean": ic["mean"], "n_references": ic["n_references"], "per_reference": refs,
            "eset_reference_mode": d["eset_reference"]["mode"], "bank_sha256": d["bank_sha256"],
            "reencode": d["reencode"], "reader_commit": d["reader_commit"], "schema": d["schema"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "purpose"))
    ap.add_argument("--archive", default=str(Path.home() / "gen3ai_archive" / "x5ab_look3_2026-10-07" / "purpose"))
    a = ap.parse_args()
    src, arc = Path(a.dir), Path(a.archive)
    runs, files = {}, {}
    for arm in ("blob", "fm"):
        for s in SEEDS:
            p = src / f"{arm}_s{s}.json"
            if not p.is_file():
                p = arc / p.name
            files[p.name] = sha(p)
            runs[f"{arm}_s{s}"] = compact(json.loads(p.read_text()))
    txt = (src / "infer.txt").read_text()
    infer = json.loads(txt[txt.index("\n{") + 1:])
    out = {"what": "X5 A/B look 3 purpose metric (1) and the reported purpose metrics; on-pool; Lane S bank",
           "gate_metric": "intent_logloss_conditional", "boundary": 1.874, "df": 14,
           "infer_text": txt[:txt.index("\n{")], "infer": infer, "runs": runs,
           "full_reads_archived_at": str(arc), "full_read_sha256": files}
    (src / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    arc.mkdir(parents=True, exist_ok=True)
    for name in files:                       # the 2.7 MB reads leave the repo; the E_row sets stay (86 KB each)
        if (src / name).is_file():
            shutil.move(str(src / name), str(arc / name))
    print(out["infer_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
