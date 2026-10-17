"""(c) STRATEGIC collapse / cycling — a read-only series from X26's own eval records (no games played).

Per eval cycle of `rb_x26_s2601/eval_results.jsonl`: the greedy, symmetric-team win rate against each frozen sentinel
(100 games each), the step gap to that sentinel, the mean bot win rate, and the in-process Hodge read's width (the
incident-restricted live scalar, the weak counterpart of `python -m main.elo`'s full decomposition).

    python designs/research_state/measurements/collapse_pilot_2026-10-17/strategic_series.py <eval_results.jsonl> <out.json>
"""
import json
import sys


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out = []
    for r in rows:
        sents = r.get("sentinels") or []
        bots = r.get("bots") or {}
        h = r.get("hodge") or {}
        out.append({
            "step": r["step"],
            "bots_mean": round(sum(bots.values()) / len(bots), 4) if bots else None,
            "sentinels": [{"gap_M": round((r["step"] - s["step"]) / 1e6, 1), "wr": s["win_rate"],
                           "n": s["counts"][1]} for s in sents],
            "sent_mean": round(sum(s["win_rate"] for s in sents) / len(sents), 4) if sents else None,
            "sent_min": min((s["win_rate"] for s in sents), default=None),
            "hodge_width_elo": h.get("width_elo"), "hodge_p": h.get("p_value"),
        })
    with open(dst, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for o in out[:: max(1, len(out) // 20)] + out[-1:]:
        print(o["step"] // 1_000_000, "M bots", o["bots_mean"], "sent mean", o["sent_mean"], "min", o["sent_min"],
              "gaps", [(s["gap_M"], s["wr"]) for s in o["sentinels"]], "hodge", o["hodge_width_elo"], o["hodge_p"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
