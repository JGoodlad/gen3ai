"""X35-frame pack-regime audit (2026-10-27): which training teams did a PRE-d8d674b9 packer play differently from the
team Showdown plays from the same paste, what does each difference DO in a battle, and what share of each arm's
training DRAWS carried one.

Two stages. Stage ``pack`` runs under the OLD checkout's ``src`` on PYTHONPATH (``git archive <pin> src``) and packs
every team exactly as that pin's training path did (``fix_gen3_hp_ivs`` then ``join_team``; the pool through that
pin's own ``TeamLoader``). Stage ``judge`` runs under main and asks main's ``showdown_pack.js`` (Showdown's
import + validator + ``Teams.pack``, ``gen3_peer_showdown_pack_v1``) for the per-field differences, classifies each
by its battle effect and prices draw shares by the arms' recorded ``--team-mix``.

usage (from the main checkout, env interpreter):
  python pack_regime_audit.py pack  <old_src_dir> <out_packs.json>        # PYTHONPATH=<old_src_dir> (set by caller)
  python pack_regime_audit.py judge <packs.json> <out_result.json>       # PYTHONPATH=<main>/src
Inputs read: data/teams (the pool; unchanged since 9c033489), src/utils/team_corpora/<corpus>, and the forum72 panel
at /home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/teams_forum72 (its sha256 list is recorded in the output).
"""
import collections
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/goodlad/dev/gen3ai")
FORUM72 = "/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/teams_forum72"
# The evaluation panels OUR side pilots in the Kakuna reads (forum72 is F; the others are judged for completeness).
PANELS = {"forum72": FORUM72,
          "home72": str(REPO / "designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72"),
          "p72": "/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/teams_p72",
          "heldout72_hpfix": "/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/teams_heldout72_hpfix",
          "rc72b": str(REPO / "designs/research_state/measurements/hctx_rc72_2026-10-26/teams_rc72b")}
CORPORA = {"w_forum_v1": "ABC", "d3_setmut_v1": "ALL", "n1_proc_v1": "ALL"}
# The arms' recorded argv (metadata.json original_command): corpus, share; sides "both" for every mix arm.
ARMS = {"C0 / LR-F / X26c / X48 (no mix)": (None, 0.0), "D1 (w_forum_v1@ABC:0.5)": ("w_forum_v1", 0.5),
        "D3 (d3_setmut_v1@ALL:0.3)": ("d3_setmut_v1", 0.3), "N1-P (n1_proc_v1@ALL:0.5)": ("n1_proc_v1", 0.5)}
TRAINEE_BIAS = 0.1   # DEFAULT_TRAINEE_BIAS_PROB: the trainee's pool draw takes a sample team 10 % of the time


def _corpus_rows(name):
    import gzip
    d = REPO / "src/utils/team_corpora" / name
    man = json.loads((d / "manifest.json").read_text())
    rows = [json.loads(ln) for ln in gzip.decompress((d / "teams.jsonl.gz").read_bytes()).decode().splitlines() if ln]
    n = int(man["tiers"][CORPORA[name]]["n"])
    return [(str(r["id"]), str(r["text"])) for r in rows[:n]]


def stage_pack(old_src, out):
    sys.path.insert(0, old_src)
    from utils.gen3_utils import fix_gen3_hp_ivs
    from utils.teambuilder import Gen3Teambuilder
    from utils.team_loader import TeamLoader
    tb = object.__new__(Gen3Teambuilder)
    loader = TeamLoader(str(REPO / "data" / "teams"))
    samples = [t for t in loader.get_sample_teams()]
    pool = loader.get_all_teams()
    src = {"pool": [(f"pool_{i}", t) for i, t in enumerate(pool)]}
    for name, d in PANELS.items():
        src[name] = [(os.path.basename(p), open(p).read()) for p in sorted(glob.glob(d + "/*.txt"))]
    for c in CORPORA:
        src[c] = _corpus_rows(c)
    res = {"old_src": old_src, "n_sample": len(samples),
           "sample_sha": sorted(hashlib.sha256(t.encode()).hexdigest() for t in samples)}
    for name, items in src.items():
        rows = []
        for i, t in items:
            try:
                p = tb.join_team(fix_gen3_hp_ivs(tb.parse_showdown_team(t)))
            except Exception as e:  # recorded, never dropped silently
                p = None
            rows.append({"id": i, "paste": t, "legacy": p, "sha": hashlib.sha256(t.encode()).hexdigest()})
        res[name] = rows
    Path(out).write_text(json.dumps(res))


def _mon_blocks(paste):
    return [b for b in paste.strip().replace("\r", "").split("\n\n") if b.strip()]


MATERIAL = {"frustration_bp", "return_bp", "ivs_hp_type_changed"}
_HP_TYPES = ["Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", "Fire", "Water", "Grass",
             "Electric", "Psychic", "Ice", "Dragon", "Dark"]


def _hp(ivs_csv):
    """Gen 3 Hidden Power (type, power) of packed-order IVs hp,atk,def,spa,spd,spe (Showdown's gen-3 formula: the
    bit order is hp, atk, def, spe, spa, spd)."""
    hp, atk, de, spa, spd, spe = (int(x) for x in ivs_csv.split(","))
    order = (hp, atk, de, spe, spa, spd)
    t = sum(((v & 1) << i) for i, v in enumerate(order)) * 15 // 63
    pw = sum((((v >> 1) & 1) << i) for i, v in enumerate(order)) * 40 // 63 + 30
    return _HP_TYPES[t], pw


def _classify(paste, diff):
    """One legacy diff line ``mon k (species) <field>: legacy X, now Y`` -> its battle effect."""
    import re
    m = re.match(r"mon (\d+) \(([^)]*)\) (\w+):", diff)
    if not m:
        return "other"
    k, field = int(m.group(1)), m.group(3)
    blocks = _mon_blocks(paste)
    moves = [ln.strip()[1:].strip().lower() for ln in (blocks[k - 1].splitlines() if k - 1 < len(blocks) else [])
             if ln.strip().startswith("-")]
    if field == "happiness":
        if "frustration" in moves:
            return "frustration_bp"      # author: happiness 0 -> BP 102; legacy 255 -> BP 1
        if "return" in moves:
            return "return_bp"           # author: happiness 0 -> BP 1; legacy 255 -> BP 102
        return "happiness_inert"         # no gen-3 battle effect without Return / Frustration
    if field == "ivs":
        if any(mv.replace(" ", "").startswith("hiddenpower") for mv in moves):
            iv = re.search(r"legacy ([\d,]+), now ([\d,]+)", diff)
            old_hp, new_hp = (_hp(iv.group(1)), _hp(iv.group(2))) if iv else (None, None)
            if old_hp is not None and old_hp == new_hp:
                return "ivs_hp_same_type"  # the same Hidden Power type and power; <= 1 point on the swapped stats
            return "ivs_hp_type_changed"   # the mon played a DIFFERENT Hidden Power than the authored one
        return "ivs_stat_only"           # a `Hidden Power: <Type>` line without the move: <= 1 point on <= 4 stats
    return "other:" + field


def stage_judge(packs_path, out):
    packs = json.loads(Path(packs_path).read_text())
    sample_sha = set(packs["sample_sha"])
    result = {"old_src": packs["old_src"], "n_sample": packs["n_sample"], "sets": {}}
    for name in ["pool", *PANELS, *CORPORA]:
        rows = packs[name]
        r = subprocess.run(["node", str(REPO / "src/utils/bridge/showdown_pack.js")], capture_output=True, text=True,
                           input=json.dumps({"format": "gen3ou", "rows": [{"paste": x["paste"], "legacy": x["legacy"]}
                                                                         for x in rows]}), check=True)
        sd = json.loads(r.stdout)
        teams = {}
        cls_teams = collections.Counter()
        cls_mons = collections.Counter()
        for x, s in zip(rows, sd):
            if x["legacy"] is None:
                teams[x["id"]] = {"error": "legacy pack raised"}
                continue
            if s["errors"] or not s["legacy_diffs"]:
                continue
            classes = [_classify(x["paste"], d) for d in s["legacy_diffs"]]
            teams[x["id"]] = {"diffs": s["legacy_diffs"], "classes": classes, "sample": x["sha"] in sample_sha}
            for c in set(classes):
                cls_teams[c] += 1
            for c in classes:
                cls_mons[c] += 1
        material = sum(1 for t in teams.values() if set(t.get("classes", ())) & MATERIAL)
        result["sets"][name] = {
            "teams": len(rows), "showdown_invalid": sum(1 for s in sd if s["errors"]),
            "legacy_pack_raised": sum(1 for x in rows if x["legacy"] is None),
            "teams_differing": sum(1 for t in teams.values() if "diffs" in t), "teams_material": material,
            "teams_by_class": dict(cls_teams), "mons_by_class": dict(cls_mons),
            "samples_differing": sum(1 for t in teams.values() if t.get("sample")),
            "samples_material": sum(1 for t in teams.values() if t.get("sample") and set(t["classes"]) & MATERIAL),
            "differing": teams,
            "input_sha256": hashlib.sha256("".join(x["sha"] for x in rows).encode()).hexdigest()}
    result["draws"] = _draw_shares(result)
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({"sets": {k: {kk: vv for kk, vv in v.items() if kk != "differing"} for k, v in result["sets"].items()},
                      "draws": result["draws"]}, indent=1))


def _draw_shares(res):
    """Per arm, the share of fresh TEAM DRAWS that is an affected team (any diff) and a MATERIAL one, per side, and
    the share of battles with at least one such team. Trainee: s * corpus + (1 - s) * (0.9 * pool + 0.1 * sample);
    opponent: s * corpus + (1 - s) * pool (team_curriculum.md "The team mix")."""
    p = res["sets"]["pool"]
    n_s = res["n_sample"]
    out = {}
    for arm, (corpus, s) in ARMS.items():
        row = {}
        for kind, key_p, key_s in (("any", "teams_differing", "samples_differing"),
                                   ("material", "teams_material", "samples_material")):
            a_pool = p[key_p] / p["teams"]
            a_samp = p[key_s] / n_s if n_s else 0.0
            a_corp = (res["sets"][corpus][key_p] / res["sets"][corpus]["teams"]) if corpus else 0.0
            tr = s * a_corp + (1 - s) * ((1 - TRAINEE_BIAS) * a_pool + TRAINEE_BIAS * a_samp)
            op = s * a_corp + (1 - s) * a_pool
            row[kind] = {"trainee": round(tr, 5), "opponent": round(op, 5), "battle_either": round(1 - (1 - tr) * (1 - op), 5)}
        out[arm] = row
    return out


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        stage_pack(sys.argv[2], sys.argv[3])
    else:
        stage_judge(sys.argv[2], sys.argv[3])
