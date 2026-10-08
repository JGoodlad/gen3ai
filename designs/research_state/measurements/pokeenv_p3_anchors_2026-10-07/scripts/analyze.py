"""P3's matched comparison, read: per cell, the two paths' win counts, the shift with its Newcombe
95% CI, the per-battle action identity (each side's CHOOSE/FORCELOSE sequence and every per-side
chunk, from the front end's captures), and every protocol / parse failure counter.

Usage: analyze.py <gate dir> [<out json>]   (cells are the subdirectories holding new/ and old/)"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("GEN3AI_SRC", "src"))
from main.anchors.results import newcombe, wilson  # noqa: E402


def load_caps(d):
    out = []
    for p in glob.glob(os.path.join(d, "*.json")):
        rec = json.load(open(p))
        out.append((int(rec["tag"].rsplit("-", 1)[1]), rec))
    return [r for _, r in sorted(out, key=lambda t: t[0])]


def side_streams(cmds):
    norm = []
    for c in cmds:
        if c.startswith("START "):
            d = json.loads(c[6:])
            d.pop("core_obs", None)
            c = "START " + json.dumps(d, sort_keys=True)
        norm.append(c)
    return {s: [c for c in norm if c.split(" ")[1:2] == [s]] for s in ("p1", "p2")}, \
        [c for c in norm if c.startswith("START")]


def errors_in(path):
    if not os.path.exists(path):
        return None
    return sum(1 for ln in open(path, errors="replace") if re.search(r"\b(ERROR|CRITICAL)\b", ln))


def cell(d):
    s_new = json.load(open(os.path.join(d, "new", "summary.json")))
    s_old = json.load(open(os.path.join(d, "old", "summary.json")))
    rows = {k: [json.loads(x) for x in open(os.path.join(d, k, "games.jsonl"))] for k in ("new", "old")}
    caps_new, caps_old = load_caps(os.path.join(d, "new", "cap")), load_caps(os.path.join(d, "old", "cap"))
    ident_actions = ident_chunks = 0
    n_choices = 0
    diffs = []
    for i, (a, b) in enumerate(zip(caps_new, caps_old)):
        sa, sta = side_streams(a["commands"])
        sb, stb = side_streams(b["commands"])
        n_choices += len(sa["p1"]) + len(sa["p2"])
        if sa == sb and sta == stb:
            ident_actions += 1
        else:
            diffs.append(i)
        if a["chunks"] == b["chunks"]:
            ident_chunks += 1
    kn, nn = s_new["wins"], s_new["n"]
    ko, no = s_old["wins"], s_old["n"]
    dlt, lo, hi = newcombe(kn, nn, ko, no)
    per_game_same = sum(1 for x, y in zip(rows["new"], rows["old"])
                        if (x["result"], x["turns"], x["our_team"]) == (y["result"], y["turns"], y["our_team"]))
    return {
        "cell": os.path.basename(d),
        "new": {"our_transport": s_new["cell"].get("our_transport"), "status": s_new["status"],
                "W/L/T": f"{s_new['wins']}/{s_new['losses']}/{s_new['ties']}", "n": nn,
                "win_rate": wilson(kn, nn), "server_log_errors": errors_in(os.path.join(d, "new", "ws_frontend.log")),
                "n_defaults": sum(r["n_defaults"] or 0 for r in rows["new"]),
                "n_redecides": sum(r["n_redecides"] or 0 for r in rows["new"]),
                "our_argmax_match_rate": s_new.get("our_argmax_match_rate"),
                "their_argmax_match_rates": s_new.get("their_argmax_match_rates"),
                "forfeits": s_new.get("hit_forfeit_limit"), "failure": s_new.get("failure")},
        "old": {"our_transport": s_old["cell"].get("our_transport"), "status": s_old["status"],
                "W/L/T": f"{s_old['wins']}/{s_old['losses']}/{s_old['ties']}", "n": no,
                "win_rate": wilson(ko, no), "server_log_errors": errors_in(os.path.join(d, "old", "ws_frontend.log")),
                "n_defaults": sum(r["n_defaults"] or 0 for r in rows["old"]),
                "n_redecides": sum(r["n_redecides"] or 0 for r in rows["old"]),
                "our_argmax_match_rate": s_old.get("our_argmax_match_rate"),
                "their_argmax_match_rates": s_old.get("their_argmax_match_rates"),
                "forfeits": s_old.get("hit_forfeit_limit"), "failure": s_old.get("failure")},
        "shift_new_minus_old": {"delta": dlt, "ci95": [lo, hi],
                                "verdict": "NOT DETECTED" if lo <= 0 <= hi else "DETECTED"},
        "battles_compared": min(len(caps_new), len(caps_old)),
        "battles_with_identical_actions_both_sides": ident_actions,
        "battles_with_identical_per_side_protocol": ident_chunks,
        "choices_compared": n_choices,
        "games_identical_result_turns_team": per_game_same,
        "diff_battles": diffs,
    }


if __name__ == "__main__":
    root = sys.argv[1]
    cells = [cell(d) for d in sorted(glob.glob(os.path.join(root, "*")))
             if os.path.isdir(os.path.join(d, "new")) and os.path.exists(os.path.join(d, "new", "summary.json"))
             and os.path.exists(os.path.join(d, "old", "summary.json"))]
    tot_n = sum(c["new"]["n"] for c in cells)
    kn = sum(int(c["new"]["W/L/T"].split("/")[0]) for c in cells)
    ko = sum(int(c["old"]["W/L/T"].split("/")[0]) for c in cells)
    no = sum(c["old"]["n"] for c in cells)
    pooled = {"n_new": tot_n, "wins_new": kn, "n_old": no, "wins_old": ko,
              "shift": newcombe(kn, tot_n, ko, no),
              "battles_identical": sum(c["battles_with_identical_actions_both_sides"] for c in cells),
              "battles": sum(c["battles_compared"] for c in cells),
              "choices": sum(c["choices_compared"] for c in cells)}
    out = {"cells": cells, "pooled": pooled}
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text + "\n")
