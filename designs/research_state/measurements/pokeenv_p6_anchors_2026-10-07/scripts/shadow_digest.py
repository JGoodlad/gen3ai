"""Digest of the `--server node` SHADOW read (legacy_shadow.py): per half and in total, the decisions compared,
the stall forfeits closed, and every record whose `diff` is non-empty or whose kind is not `decision`
(a reader halt, a frame race), with the read's own protocol / failure counters beside them.

Usage: shadow_digest.py <read dir> [<out json>]"""
import collections
import json
import os
import re
import sys


def digest(d):
    out = {"halves": {}, "total": collections.Counter(), "examples": []}
    for half in ("ours_challenge", "peer_challenge"):
        p = os.path.join(d, half, "shadow.jsonl")
        c = collections.Counter()
        if os.path.exists(p):
            for ln in open(p):
                r = json.loads(ln)
                kind = r.get("kind")
                if kind != "decision":
                    c[f"kind:{kind}"] += 1
                    if len(out["examples"]) < 12:
                        out["examples"].append({"half": half, **r})
                    continue
                c["decisions"] += 1
                if r.get("stall_forfeit"):
                    c["stall_forfeits"] += 1
                elif r.get("sent", "").startswith("/forfeit"):
                    c["forfeits_after_embed"] += 1
                else:
                    c["choices_compared"] += 1
                for f in r.get("diff") or []:
                    c[f"diff:{f}"] += 1
                if r.get("diff") and len(out["examples"]) < 12:
                    out["examples"].append({"half": half, **r})
        out["halves"][half] = dict(c)
        out["total"].update(c)
    out["total"] = dict(out["total"])
    s = json.load(open(os.path.join(d, "summary.json")))
    log = os.path.join(d, "showdown.log")
    out["read"] = {"status": s["status"], "n": s["n"], "W/L/T": f"{s['wins']}/{s['losses']}/{s['ties']}",
                   "our_transport": s["cell"].get("our_transport"), "server_impl": s["cell"].get("server_impl"),
                   "server_version": s["cell"].get("server_version"), "failure": s.get("failure"),
                   "forfeits": s.get("hit_forfeit_limit"),
                   "their_argmax_match_rates": s.get("their_argmax_match_rates"),
                   "server_log_errors": (sum(1 for ln in open(log, errors="replace")
                                             if re.search(r"\b(ERROR|CRITICAL)\b", ln))
                                         if os.path.exists(log) else None)}
    rows = [json.loads(x) for x in open(os.path.join(d, "games.jsonl"))]
    out["read"]["n_defaults"] = sum(r["n_defaults"] or 0 for r in rows)
    out["read"]["n_redecides"] = sum(r["n_redecides"] or 0 for r in rows)
    out["read"]["turns"] = [r["turns"] for r in rows]
    return out


if __name__ == "__main__":
    res = digest(sys.argv[1])
    text = json.dumps(res, indent=1, default=str)
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text + "\n")
