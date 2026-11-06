"""S'-HONEST design input 1 (2026-11-05, OUTCOME-BLIND): what S'-LIVE's searched cells' (every final attempt dir; the 9 `.partial.<epoch>` dirs a resume set aside are skipped) decision records say
about two leak channels and the search's CPU split. Reads only ``search_decisions.jsonl`` (never ``games.jsonl`` or a
summary's result).

* the template's leak block on every search-path decision (searched or excluded): ``attribution_miss`` (a past
  opponent move token kept VERBATIM) and ``rewritten_moves`` / ``rewritten_switches`` (a past opponent choice the
  searched side never saw executed, rendered at the TRUE slot);
* on the hashed re-score sample (it keeps the honest log): which opponent move tokens are absent from the TRUE set;
* the CPU split: ``cpu_s`` of searched and excluded decisions, ``world_cpu_s`` / ``playout_cpu_s`` of searched ones.

    python3 decision_records.py <S'-LIVE state>/anchor/u/pstar_s/cells decision_records.json
"""
import collections
import glob
import json
import re
import sys


def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main(cells, out):
    n = srch = excl = 0
    hist = {k: collections.Counter() for k in ("attribution_miss", "rewritten_moves", "rewritten_switches")}
    miss_tok = collections.Counter()
    sampled = 0
    cpu = {"searched": [], "world": [], "playout": [], "excluded": []}
    for f in sorted(f for f in glob.glob(cells + "/*/search_decisions.jsonl") if ".partial." not in f):
        for line in open(f):
            d = json.loads(line)
            if d.get("why") in ("before_turn", "one_legal"):
                continue
            n += 1
            if d.get("searched"):
                srch += 1
                cpu["searched"].append(d["cpu_s"])
                cpu["world"].append(d["world_cpu_s"])
                cpu["playout"].append(d["playout_cpu_s"])
            else:
                excl += 1
                cpu["excluded"].append(d["cpu_s"])
            lk = d.get("leak") or {}
            for k in hist:
                hist[k][min(int(lk.get(k, 0)), 5)] += 1
            if d.get("rescore_sample"):
                sampled += 1
                log = d["log"]
                oi = 1 if d["side"] == "p1" else 0
                opp = "p2" if d["side"] == "p1" else "p1"
                true_moves = {toid(m) for e in log["teams"][oi].split("]") for m in e.split("|")[4].split(",")}
                for c in log["cmds"]:
                    p = c.split(" ", 3)
                    if len(p) == 4 and p[1] == opp and p[2] == "move" and toid(p[3]) not in true_moves:
                        miss_tok[toid(p[3])] += 1
    share = {k: {"ge1": round(sum(v for b, v in h.items() if b >= 1) / n, 4), "hist_capped5": dict(sorted(h.items()))}
             for k, h in hist.items()}
    tot = {k: round(sum(v) / 3600, 1) for k, v in cpu.items()}
    rec = {"search_path_decisions": n, "searched": srch, "excluded": excl, "leak_block": share,
           "sampled_logs": sampled, "opponent_move_tokens_absent_from_true_set": dict(miss_tok.most_common()),
           "cpu_s_mean": {k: round(sum(v) / max(1, len(v)), 2) for k, v in cpu.items()},
           "cpu_h_total": tot}
    json.dump(rec, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
