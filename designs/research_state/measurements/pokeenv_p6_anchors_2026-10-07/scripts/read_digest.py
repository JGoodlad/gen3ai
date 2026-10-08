"""One compact line per anchor read: status, W/L/T, our reader, the server, every failure / protocol counter, our
decision count, the forfeits, the peer's verification. Usage: read_digest.py <read dir>... [--json <out>]"""
import json
import os
import re
import sys


def digest(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    rows = [json.loads(x) for x in open(os.path.join(d, "games.jsonl"))]
    logs = [p for p in ("ws_frontend.log", "showdown.log") if os.path.exists(os.path.join(d, p))]
    errs = {p: sum(1 for ln in open(os.path.join(d, p), errors="replace") if re.search(r"\b(ERROR|CRITICAL)\b", ln))
            for p in logs}
    return {"read": os.path.basename(d.rstrip("/")), "status": s["status"], "n": s["n"],
            "W/L/T": f"{s['wins']}/{s['losses']}/{s['ties']}", "our_transport": s["cell"].get("our_transport"),
            "our_side": s["cell"].get("our_side"), "server_impl": s["cell"].get("server_impl"),
            "server_version": s["cell"].get("server_version"), "failure": s.get("failure"),
            "our_decisions": sum(r["n_decisions"] or 0 for r in rows),
            "n_defaults": sum(r["n_defaults"] or 0 for r in rows),
            "n_redecides": sum(r["n_redecides"] or 0 for r in rows),
            "hit_forfeit_limit": sum(1 for r in rows if r["hit_forfeit_limit"]), "max_turns": max(r["turns"] for r in rows),
            "regime_verified_decisions": s.get("regime_verified_decisions"), "peer_clean": s.get("peer_clean"),
            "their_argmax_match_rates": s.get("their_argmax_match_rates"),
            "our_argmax_match_rate": s.get("our_argmax_match_rate"),
            "our_bot_seed": s["cell"].get("our_bot_seed"), "server_log_errors": errs}


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        out = args[args.index("--json") + 1]
        args = args[:args.index("--json")]
    res = [digest(d) for d in args]
    for r in res:
        print(json.dumps(r))
    if out:
        open(out, "w").write(json.dumps(res, indent=1) + "\n")
