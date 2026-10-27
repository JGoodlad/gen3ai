"""Our packer (the checkout on PYTHONPATH) vs Showdown's reading of each paste, per played field (showdown_pack.js
legacy_diffs: pack_norm fields + gender + name). usage: pack_audit.py <checkout root>"""
import collections, json, subprocess, sys, glob, os
from pathlib import Path
root = Path(sys.argv[1])
from utils.teambuilder import Gen3Teambuilder
import utils.teambuilder as TB
from utils.team_loader import TeamLoader
from utils.team_corpora import read_dir
tb = object.__new__(Gen3Teambuilder)
fix = getattr(TB, "fix_gen3_hp_ivs")
def pack(t): return tb.join_team(fix(tb.parse_showdown_team(t)))
sources = {}
sources["forum72"] = [(os.path.basename(p), open(p).read().strip()) for p in sorted(glob.glob("/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/teams_forum72/*"))]
pool = TeamLoader(str(root / "data" / "teams")).get_all_teams()
sources["pool"] = [(f"pool_{i}", t.strip()) for i, t in enumerate(pool)]
for c in ("d3_setmut_v1", "n1_proc_v1", "w_forum_v1"):
    _, rows, _, _ = read_dir(root / "src/utils/team_corpora" / c)
    sources[c] = [(str(r["id"]), str(r["text"])) for r in rows]
out = {}
for name, items in sources.items():
    packs = []
    for _, t in items:
        try: packs.append(pack(t))
        except Exception as e: packs.append(None)
    r = subprocess.run(["node", str(root / "src/utils/bridge/showdown_pack.js")] if (root / "src/utils/bridge/showdown_pack.js").exists()
                       else ["node", "/home/goodlad/dev/gen3ai/src/utils/bridge/showdown_pack.js"],
                       capture_output=True, text=True, input=json.dumps({"format": "gen3ou", "rows": [{"paste": t, "legacy": p} for (_, t), p in zip(items, packs)]}))
    rows = json.loads(r.stdout)
    fields = collections.Counter(); bad = {}
    invalid = sum(1 for x in rows if x["errors"])
    for (i, t), x in zip(items, rows):
        if x["errors"]: continue
        if x["legacy_diffs"]:
            bad[i] = x["legacy_diffs"][:4]
            for d in x["legacy_diffs"]:
                fields[d.split(") ", 1)[-1].split(":")[0] if ") " in d else d] += 1
    raw = sum(1 for _, t in items if "Happiness" in t), sum(1 for _, t in items if "Frustration" in t), sum(1 for _, t in items if "- Return" in t)
    out[name] = {"teams": len(items), "showdown_invalid": invalid, "teams_differing": len(bad), "by_field": dict(fields),
                 "raw_happiness_frustration_return": raw, "examples": dict(list(bad.items())[:6])}
print(json.dumps(out, indent=1))
