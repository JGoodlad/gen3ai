"""Eval U2's real-launch gate, the `--promotion-sprt` half: one DECISION row per tested candidate, each
promotion batch row's counts and pair totals equal the run's own `sprt_promotion.jsonl` verdict, and the
decision row's verdict / LLR / pairs equal the verdict line.

    python check_sprt_ledger.py --ledger <root>/_ledger --run-dir <models>/<run> [--out check.json]

Reads `decisions/*.jsonl` and `rows/*/*.jsonl` (purpose `promotion`) of the ledger archive and the
append-only `sprt_promotion.jsonl` (the trainer's own log, written independently of the ledger).
Exits non-zero on any disagreement or a candidate with a verdict line and no decision row (or the
reverse).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    run = a.run_dir.rstrip("/").split("/")[-1]
    verdicts = {}
    for line in open(f"{a.run_dir}/sprt_promotion.jsonl"):
        d = json.loads(line)
        if d["event"] == "verdict":
            verdicts[int(d["step"])] = d
    decs = []
    for p in sorted(glob.glob(f"{a.ledger}/decisions/*.jsonl")):
        decs += [json.loads(line) for line in open(p) if line.strip()]
    rows = []
    for p in sorted(glob.glob(f"{a.ledger}/rows/*/*.jsonl")):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    decs = [d for d in decs if str(d.get("request_id", "")).startswith(f"{run}:sprt:")]
    prom = [r for r in rows if r.get("run") == run and r.get("purpose") == "promotion"]
    bad = 0
    out = {"run": run, "verdict_lines": len(verdicts), "decision_rows": len(decs), "promotion_rows": len(prom), "candidates": {}}
    steps_with_dec = set()
    for d in decs:
        rid = str(d["request_id"])
        step = int(rid.rsplit(":", 1)[-1])
        out.setdefault("decision_keys", sorted(d.keys()))
        if step is None:
            continue
        steps_with_dec.add(step)
        v = verdicts.get(step)
        c = {"decision_row": {k: d[k] for k in ("decision_id", "kind", "rule", "verdict", "request_id")},
             "consumed_rows": d["consumed"]["count"]}
        # batch rows of this candidate: pair totals = sum of counts
        mine = [r for r in prom if str(r["request"]["id"]) == rid]
        games = sum(r["counts"]["w"] + r["counts"]["l"] + r["counts"]["d"] for r in mine)
        c["promotion_rows"] = len(mine)
        c["games_in_rows"] = games
        c["verdict_line"] = {k: v.get(k) for k in ("verdict", "reason", "n_pairs", "llr", "promoted", "pair_counts")} if v else None
        ok = v is not None and games == 2 * int(v["n_pairs"]) and d["verdict"] == v["verdict"] \
            and d["consumed"]["count"] == len(mine) and d["kind"] == "promotion"
        wins = sum(r["counts"]["w"] for r in mine)
        n = sum(r["counts"]["w"] + r["counts"]["l"] + r["counts"]["d"] for r in mine)
        c["score_from_rows"] = round((wins + 0.5 * sum(r["counts"]["d"] for r in mine)) / n, 6) if n else None
        c["score_from_verdict_line"] = v.get("test_score_selected") if v else None
        ok = ok and v is not None and c["score_from_rows"] is not None and abs(c["score_from_rows"] - v["test_score_selected"]) < 1e-5
        c["ok"] = bool(ok)
        bad += 0 if ok else 1
        out["candidates"][str(step)] = c
        print(f"candidate {step:>9,d}: decision rows ok={ok}; promotion rows {len(mine)}, games {games} == 2 x {v['n_pairs'] if v else '?'} pairs; "
              f"score rows {c['score_from_rows']} vs verdict line {c['score_from_verdict_line']}; verdict {c['verdict_line']}")
    missing = sorted(set(verdicts) - steps_with_dec)
    extra = sorted(steps_with_dec - set(verdicts))
    if missing or extra:
        print(f"verdict lines without a decision row: {missing}; decision rows without a verdict line: {extra}"); bad += 1
    n_per = {s: sum(1 for d in decs if str(d["request_id"]).endswith(f":sprt:{s}")) for s in verdicts}
    out["decision_rows_per_candidate"] = n_per
    if any(v != 1 for v in n_per.values()):
        print(f"not exactly one decision row per candidate: {n_per}"); bad += 1
    out["problems"] = bad
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"candidates with a verdict: {len(verdicts)}; decision rows: {len(decs)}; per candidate: {n_per}; problems: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
