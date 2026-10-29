"""X63 union pack-regime audit (2026-10-29): ``x35_pack_regime_2026-10-27/pack_regime_audit.py``'s METHOD on
``x63_union_v1@ALL`` -- which union teams the pre-``d8d674b9`` packer (the production pin ``f4b25238``) plays
differently from the team Showdown plays from the same paste, classified by battle effect (MATERIAL = Frustration /
Return base power or a changed Hidden Power type). STATUS §7 risk 4 / that audit's condition (a): above 2 % of a new
corpus's teams, the arm needs a packer-backport pin.

Two stages, as the original:
  PYTHONPATH=<old_src>         python union_pack_regime_audit.py pack  <old_src> <union_dir> <out_packs.json>
  PYTHONPATH=<worktree>/src    python union_pack_regime_audit.py judge <packs.json> <out_result.json>
``<old_src>`` is ``git archive f4b25238 src`` unpacked; ``pack`` packs every union team exactly as that pin's training
path did (``fix_gen3_hp_ivs`` then ``join_team``); ``judge`` asks this checkout's ``showdown_pack.js`` (Showdown's
import + validator + ``Teams.pack``) for the per-field differences and classifies them with the original audit's own
``_classify`` (imported, not copied). CPU only.
"""
import collections
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent.parent / "x35_pack_regime_2026-10-27" / "pack_regime_audit.py"


def _union_rows(union_dir):
    d = Path(union_dir)
    man = json.loads((d / "manifest.json").read_text())
    raw = (d / "teams.jsonl.gz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != man["data_sha256"]:
        raise SystemExit(f"{d}: data sha256 differs from its manifest")
    rows = [json.loads(ln) for ln in gzip.decompress(raw).decode().splitlines() if ln]
    return man, rows


def stage_pack(old_src, union_dir, out):
    sys.path.insert(0, old_src)
    from utils.gen3_utils import fix_gen3_hp_ivs
    from utils.teambuilder import Gen3Teambuilder

    tb = object.__new__(Gen3Teambuilder)
    man, rows = _union_rows(union_dir)
    res = {"old_src": old_src, "union": {"name": man["name"], "data_sha256": man["data_sha256"],
                                         "tier_sha256": man["tiers"]["ALL"]["sha256"]}, "rows": []}
    for r in rows:
        try:
            p = tb.join_team(fix_gen3_hp_ivs(tb.parse_showdown_team(r["text"])))
        except Exception as e:  # recorded, never dropped silently
            p = None
            r = dict(r, pack_error=f"{type(e).__name__}: {e}"[:300])
        res["rows"].append({"id": r["id"], "curator": r["curator"], "paste": r["text"], "legacy": p,
                            **({"pack_error": r["pack_error"]} if "pack_error" in r else {})})
    Path(out).write_text(json.dumps(res))


def stage_judge(packs_path, out):
    spec = importlib.util.spec_from_file_location("pack_regime_audit", ORIGINAL)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    from utils.paths import src_path

    packs = json.loads(Path(packs_path).read_text())
    rows = packs["rows"]
    sd = []
    for k in range(0, len(rows), 3000):
        chunk = rows[k:k + 3000]
        r = subprocess.run(["node", str(src_path("utils", "bridge", "showdown_pack.js"))], capture_output=True,
                           text=True, check=True,
                           input=json.dumps({"format": "gen3ou", "rows": [{"paste": x["paste"], "legacy": x["legacy"]}
                                                                         for x in chunk]}))
        sd += json.loads(r.stdout)
    if len(sd) != len(rows):
        raise SystemExit(f"showdown_pack.js answered {len(sd)} of {len(rows)}")
    comps = sorted({x["curator"] for x in rows}, key=[x["curator"] for x in rows].index)
    per = {c: {"teams": 0, "showdown_invalid": 0, "legacy_pack_raised": 0, "teams_differing": 0, "teams_material": 0,
               "teams_by_class": collections.Counter(), "mons_by_class": collections.Counter()} for c in comps}
    differing = {}
    for x, s in zip(rows, sd):
        c = per[x["curator"]]
        c["teams"] += 1
        if x["legacy"] is None:
            c["legacy_pack_raised"] += 1
            differing[x["id"]] = {"curator": x["curator"], "error": x.get("pack_error", "legacy pack raised")}
            continue
        if s["errors"]:
            c["showdown_invalid"] += 1
            continue
        if not s["legacy_diffs"]:
            continue
        classes = [A._classify(x["paste"], d) for d in s["legacy_diffs"]]
        c["teams_differing"] += 1
        if set(classes) & A.MATERIAL:
            c["teams_material"] += 1
        for k in set(classes):
            c["teams_by_class"][k] += 1
        for k in classes:
            c["mons_by_class"][k] += 1
        differing[x["id"]] = {"curator": x["curator"], "diffs": s["legacy_diffs"], "classes": classes}
    n = len(rows)
    tot = {k: sum(per[c][k] for c in comps) for k in ("teams", "showdown_invalid", "legacy_pack_raised",
                                                      "teams_differing", "teams_material")}
    result = {
        "method": "x35_pack_regime_2026-10-27/pack_regime_audit.py (pack under the old pin's src, judge with "
                  "showdown_pack.js, its _classify); MATERIAL = " + ", ".join(sorted(A.MATERIAL)),
        "old_src": packs["old_src"], "union": packs["union"],
        "total": dict(tot, share_material=round(tot["teams_material"] / n, 5),
                      share_differing=round(tot["teams_differing"] / n, 5),
                      threshold_share=0.02, above_threshold=tot["teams_material"] / n > 0.02),
        "by_component": {c: dict(v, teams_by_class=dict(v["teams_by_class"]), mons_by_class=dict(v["mons_by_class"]),
                                 share_material=round(v["teams_material"] / v["teams"], 5)) for c, v in per.items()},
        "material_ids": sorted(i for i, d in differing.items() if set(d.get("classes", ())) & A.MATERIAL),
        "differing": differing,
    }
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("total", "by_component")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        stage_pack(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        stage_judge(sys.argv[2], sys.argv[3])
