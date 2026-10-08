"""P6's matched comparison, read (P3's `analyze.py`, extended): per cell, the two paths' win counts, the shift
with its Newcombe 95% CI, the per-battle identity (each side's CHOOSE/FORCELOSE sequence, the START line and
every per-side chunk, from the front end's captures), the first difference of every differing battle, and every
protocol / parse failure counter.

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


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"at": i, "new": x, "old": y}
    if len(a) != len(b):
        return {"at": min(len(a), len(b)), "len_new": len(a), "len_old": len(b)}
    return None


def errors_in(path):
    if not os.path.exists(path):
        return None
    return sum(1 for ln in open(path, errors="replace") if re.search(r"\b(ERROR|CRITICAL)\b", ln))


def our_side_of(rec, prefix):
    """'p1' / 'p2' — the side our client played (its name starts with the anchors username)."""
    d = json.loads(rec["commands"][0][6:])
    return "p1" if d["p1"]["name"].startswith(prefix) else "p2"


def cell(d, prefix="Gen3AIAnchor"):
    s_new = json.load(open(os.path.join(d, "new", "summary.json")))
    s_old = json.load(open(os.path.join(d, "old", "summary.json")))
    rows = {k: [json.loads(x) for x in open(os.path.join(d, k, "games.jsonl"))] for k in ("new", "old")}
    caps_new, caps_old = load_caps(os.path.join(d, "new", "cap")), load_caps(os.path.join(d, "old", "cap"))
    ident_actions = ident_chunks = 0
    n_choices = n_ours = 0
    diffs = []
    for i, (a, b) in enumerate(zip(caps_new, caps_old)):
        sa, sta = side_streams(a["commands"])
        sb, stb = side_streams(b["commands"])
        n_choices += len(sa["p1"]) + len(sa["p2"])
        n_ours += len(sa[our_side_of(a, prefix)])
        same_actions = sa == sb and sta == stb
        same_chunks = a["chunks"] == b["chunks"]
        ident_actions += same_actions
        ident_chunks += same_chunks
        if not (same_actions and same_chunks):
            diffs.append({"battle": i + 1, "tag": a["tag"], "start": first_diff(sta, stb),
                          "p1": first_diff(sa["p1"], sb["p1"]), "p2": first_diff(sa["p2"], sb["p2"]),
                          "chunks_equal": same_chunks})
    kn, nn = s_new["wins"], s_new["n"]
    ko, no = s_old["wins"], s_old["n"]
    dlt, lo, hi = newcombe(kn, nn, ko, no)

    def side(s, k):
        return {"our_transport": s["cell"].get("our_transport"), "status": s["status"],
                "W/L/T": f"{s['wins']}/{s['losses']}/{s['ties']}", "n": s["n"], "win_rate": wilson(s["wins"], s["n"]),
                "server_log_errors": errors_in(os.path.join(d, k, "ws_frontend.log")),
                "n_defaults": sum(r["n_defaults"] or 0 for r in rows[k]),
                "n_redecides": sum(r["n_redecides"] or 0 for r in rows[k]),
                "our_argmax_match_rate": s.get("our_argmax_match_rate"),
                "their_argmax_match_rates": s.get("their_argmax_match_rates"),
                "forfeits": s.get("hit_forfeit_limit"), "failure": s.get("failure"),
                "our_bot_seed": s["cell"].get("our_bot_seed")}

    return {
        "cell": os.path.basename(d),
        "new": side(s_new, "new"),
        "old": side(s_old, "old"),
        "shift_new_minus_old": {"delta": dlt, "ci95": [lo, hi],
                                "verdict": "NOT DETECTED" if lo <= 0 <= hi else "DETECTED"},
        "battles_compared": min(len(caps_new), len(caps_old)),
        "captures": [len(caps_new), len(caps_old)],
        "battles_with_identical_actions_both_sides": ident_actions,
        "battles_with_identical_per_side_protocol": ident_chunks,
        "choices_compared": n_choices,
        "our_choices_compared": n_ours,
        "games_identical_result_turns_team": sum(
            1 for x, y in zip(rows["new"], rows["old"])
            if (x["result"], x["turns"], x["our_team"]) == (y["result"], y["turns"], y["our_team"])),
        "diffs": diffs,
    }


if __name__ == "__main__":
    root = sys.argv[1]
    cells = [cell(d) for d in sorted(glob.glob(os.path.join(root, "*")))
             if os.path.isdir(os.path.join(d, "new")) and os.path.exists(os.path.join(d, "new", "summary.json"))
             and os.path.exists(os.path.join(d, "old", "summary.json"))]
    # a TEETH cell (deliberately mismatched) is reported, never pooled
    real = [c for c in cells if not c["cell"].startswith("teeth")]
    kn = sum(int(c["new"]["W/L/T"].split("/")[0]) for c in real)
    ko = sum(int(c["old"]["W/L/T"].split("/")[0]) for c in real)
    nn = sum(c["new"]["n"] for c in real)
    no = sum(c["old"]["n"] for c in real)
    pooled = {"n_new": nn, "wins_new": kn, "n_old": no, "wins_old": ko, "shift": newcombe(kn, nn, ko, no),
              "battles_identical": sum(c["battles_with_identical_actions_both_sides"] for c in real),
              "battles_chunks_identical": sum(c["battles_with_identical_per_side_protocol"] for c in real),
              "battles": sum(c["battles_compared"] for c in real),
              "choices": sum(c["choices_compared"] for c in real),
              "our_choices": sum(c["our_choices_compared"] for c in real)}
    text = json.dumps({"cells": cells, "pooled_excluding_teeth": pooled}, indent=1, default=str)
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text + "\n")
