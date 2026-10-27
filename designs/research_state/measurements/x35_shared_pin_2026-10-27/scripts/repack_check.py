"""Does the front end's validator re-pack (ws_frontend since ce50227a5) change any set OUR side plays at the pin?
For every team file of each source: the pin's (f4b25238's) Gen3Teambuilder pack -> showdown_pack.js with paste = that
pack and legacy = that pack: `packed` is the validator's re-pack (what the front end plays), `legacy_diffs` the played
fields that differ (pack_norm fields + gender + name). Run with PYTHONPATH = the pin's src, cwd = the pin."""
import json
import subprocess
import sys
from pathlib import Path

from main.anchors.session import load_team_texts
from utils.bridge.team_validator import validate_teams_locally
from utils.teambuilder import Gen3Teambuilder

out = {}
for label, d in [a.split("=", 1) for a in sys.argv[1:]]:
    items = load_team_texts(Path(d))
    texts = [t for _, t in items]
    tb = Gen3Teambuilder(texts, rng_seed=0)
    packs = list(tb.packed_teams)
    vs = validate_teams_locally("gen3ou", texts)
    names = [n for (n, _), v in zip(items, vs) if v.get("valid")]
    assert len(names) == len(packs), (label, len(names), len(packs))
    r = subprocess.run(["node", "src/utils/bridge/showdown_pack.js"], capture_output=True, text=True, check=False,
                       input=json.dumps({"format": "gen3ou", "rows": [{"paste": p, "legacy": p} for p in packs]}))
    rows = json.loads(r.stdout)
    # the ws_frontend's own path, too: validate_team.js's `packed` of our pack
    fe = validate_teams_locally("gen3ou", packs)
    diffs = {n: row["legacy_diffs"] for n, row in zip(names, rows) if row["legacy_diffs"] or row["errors"]}
    fe_same = sum(1 for row, f in zip(rows, fe) if f.get("packed") == row["packed"])
    out[label] = {"dir": d, "files": len(items), "packed": len(packs), "string_equal": sum(p == row["packed"] for p, row in zip(packs, rows)),
                  "frontend_repack_equals_showdown_pack": fe_same, "played_diffs": diffs}
print(json.dumps(out, indent=1))
