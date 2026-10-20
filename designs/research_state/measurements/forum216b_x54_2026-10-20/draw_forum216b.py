"""forum216b -- X54's CONFIRMATION panel, drawn by the rule README.md "The draw rule" states. It plays nothing.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src nice -n 19 taskset -c 5-7,13-15 python <this> [--check]

The source is ``w_forum_v1``'s tier AB AS COMMITTED BEFORE THE CARVE (``SOURCE_*`` below: the corpus already applied
the verbatim / HP-repair / pool-exact / panel / validator / engine / duplicate / strength / curator-cap rules), read
from git at ``SOURCE_COMMIT`` so the draw reproduces after the corpus is rebuilt without these teams. Writes
``teams_forum216b/<id>.txt`` and ``forum216b_manifest.json`` beside this script; ``--check`` exits 1 unless the
committed files reproduce byte for byte. Never installs the panel anywhere a read could reach it (no Metamon cache, no
config): it is SEALED until X54 (``main.heldout_panels.SEALED``).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from main import heldout_panels as HP
from main.anchors import team_overlap as TO
from main.anchors.session import pool_team_texts
from utils.bridge.hp_set_check import check_teams
from utils.bridge.team_validator import validate_teams_locally
from utils.paths import repo_root

HERE = Path(__file__).resolve().parent
NAME = "forum216b"
SCHEMA = "x54_forum216b_manifest_v1"
SOURCE_COMMIT = "b3886800c"           # main HEAD when the carve was ordered (2026-10-20)
SOURCE_DATA = "src/utils/team_corpora/w_forum_v1/teams.jsonl.gz"
SOURCE_DATA_SHA256 = "4af495c49a2df0266c8fe10c2691cea274e683da0f58c4e82d2a0124b7a17469"
SOURCE_TIER = "AB"
SOURCE_TIER_N = 559
SOURCE_TIER_SHA256 = "fa1219f6799f0d4ab93383b93d2d12b0e5b8c9a054419f629680f0c45b681095"
SEED = 5400
#: forum72's stratum counts (CORE 4 / NEAR 54 / FAR 14 of 72, its manifest's `strata`) x 3.
QUOTA = {"CORE": 12, "NEAR": 162, "FAR": 42}
N = sum(QUOTA.values())
#: forum72's curator cap (14 per 72) x 3.
CURATOR_CAP = 42
#: Every panel a drawn team must not copy or near-copy (>= 4 shared mon sets), besides the pool and earlier draws.
REF_PANELS = ("home72", "away", "heldout72", "heldout72_hpfix", "forum72", "p72")
#: ladder_weights_v1's reference: the SAMPLE teams (home72 + away), species only.
STRATUM_REF = ("home72", "away")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stratum_of(k: int) -> str:
    """ladder_weights_v1: best SPECIES overlap with one sample team -- CORE 6, NEAR 4-5, FAR <= 3."""
    return "CORE" if k == 6 else ("NEAR" if k >= 4 else "FAR")


def stratum_fn():
    ref = [t for t in TO.parse_items([it for p in STRATUM_REF for it in HP.panel_team_texts(p)]) if len(t.mons) == 6]
    species = list({t.species6 for t in ref})
    return lambda text: stratum_of(max(len(TO.parse_team(text).species6 & s) for s in species))


def source_rows() -> List[Dict[str, Any]]:
    raw = subprocess.run(["git", "-C", str(repo_root()), "show", f"{SOURCE_COMMIT}:{SOURCE_DATA}"],
                         capture_output=True, check=True).stdout
    if _sha(raw) != SOURCE_DATA_SHA256:
        raise SystemExit(f"{SOURCE_COMMIT}:{SOURCE_DATA} sha256 {_sha(raw)} != {SOURCE_DATA_SHA256}")
    rows = [json.loads(ln) for ln in gzip.decompress(raw).decode().splitlines() if ln]
    ab = [r for r in rows if r["tier"] in ("A", "B")]
    if len(ab) != SOURCE_TIER_N or rows[:SOURCE_TIER_N] != ab:
        raise SystemExit(f"tier AB is not the first {SOURCE_TIER_N} rows")
    h = hashlib.sha256()
    for r in ab:
        h.update(r["text"].encode())
        h.update(b"\x00")
    if h.hexdigest() != SOURCE_TIER_SHA256:
        raise SystemExit(f"tier AB sha256 {h.hexdigest()} != {SOURCE_TIER_SHA256}")
    return ab


def hp_bad(verdict: Dict[str, Any]) -> list:
    """Hidden Power sets that do not play the type their line names at power 70 (forum72 / P72's rule)."""
    return [s for s in verdict.get("sets", []) if s.get("hp_move") and (
        s.get("hp_type") != s["hp_move"].split()[-1] or s.get("power") != 70
        or s.get("ivs_after_validate") != s.get("ivs_in"))]


def references() -> Dict[str, Any]:
    pool = [TO.parse_team(t, f"pool{i:03d}") for i, t in enumerate(pool_team_texts())]
    refs = {"pool": TO.Reference(pool)}
    for p in REF_PANELS:
        refs[p] = TO.Reference(TO.parse_items(HP.panel_team_texts(p)))
    return refs


def reject_reason(team: TO.Team, refs: Dict[str, Any]) -> Optional[str]:
    if len(team.mons) != 6:
        return "not 6 mons"
    for name, ref in refs.items():
        if ref.keys[team.key]:
            return f"exact team in {name}"
        k, lab = ref.max_shared(team)
        if k >= HP.NEAR_NEIGHBOUR_MIN_SHARED:
            return f"shares {k} mon sets with {name}:{lab}"
    return None


def draw() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = source_rows()
    by_id = {r["id"]: r for r in rows}
    order = sorted(by_id)
    random.Random(SEED).shuffle(order)
    strat = stratum_fn()
    refs = references()
    texts = [by_id[i]["text"] for i in order]
    verdicts = dict(zip(order, validate_teams_locally("gen3ou", texts)))
    hp = dict(zip(order, check_teams(texts)))
    supply = collections.Counter(strat(by_id[i]["text"]) for i in order)
    taken: collections.Counter = collections.Counter()
    per_cur: collections.Counter = collections.Counter()
    drawn: List[Dict[str, Any]] = []
    drawn_teams: List[TO.Team] = []
    log: List[Dict[str, Any]] = []
    for i in order:
        if len(drawn) == N:
            break
        r = by_id[i]
        s = strat(r["text"])
        if taken[s] >= QUOTA[s]:
            log.append({"id": i, "stratum": s, "skipped": "stratum full"})
            continue
        if per_cur[r["curator"]] >= CURATOR_CAP:
            log.append({"id": i, "stratum": s, "curator": r["curator"], "skipped": "curator cap"})
            continue
        team = TO.parse_team(r["text"], i)
        why = None if verdicts[i].get("valid") else f"validator: {verdicts[i].get('errors')}"
        if not why and hp_bad(hp[i]):
            why = f"hidden power mismatch: {hp_bad(hp[i])}"
        why = why or reject_reason(team, {**refs, "drawn": TO.Reference(drawn_teams)})
        if why:
            log.append({"id": i, "stratum": s, "rejected": why})
            continue
        taken[s] += 1
        per_cur[r["curator"]] += 1
        drawn.append({"id": i, "stratum": s, "tier": r["tier"], "curator": r["curator"], "text": r["text"]})
        drawn_teams.append(team)
        log.append({"id": i, "stratum": s, "accepted": len(drawn)})
    if dict(taken) != QUOTA:
        raise SystemExit(f"the draw could not fill its quotas: {dict(taken)} != {QUOTA}")
    stats = {"supply_by_stratum": dict(supply), "per_curator": dict(per_cur.most_common()),
             "by_tier": dict(collections.Counter(d["tier"] for d in drawn)),
             "by_stratum_tier": {s: dict(collections.Counter(d["tier"] for d in drawn if d["stratum"] == s))
                                 for s in QUOTA}}
    return drawn, log, stats


def overlap_check(drawn: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The registry's question after the fact: the max shared mon sets of every drawn team with each reference."""
    refs = references()
    out = {}
    for name, ref in refs.items():
        hist: collections.Counter = collections.Counter()
        for d in drawn:
            hist[ref.max_shared(TO.parse_team(d["text"]))[0]] += 1
        out[name] = {str(k): hist.get(k, 0) for k in range(7)}
    teams = [TO.parse_team(d["text"], d["id"]) for d in drawn]
    intra = max((TO.Reference(teams[:j] + teams[j + 1:]).max_shared(t)[0] for j, t in enumerate(teams)), default=0)
    return {"max_shared_mon_sets_hist": out, "intra_panel_max_shared": int(intra)}


def build() -> Tuple[Dict[str, bytes], bytes]:
    from main.anchors.session import team_set_digest

    drawn, log, stats = draw()
    files = {f"{d['id']}.txt": (d["text"] + "\n").encode() for d in drawn}
    manifest = {
        "schema": SCHEMA,
        "name": NAME,
        "what": "X54's CONFIRMATION panel (design_beat_metamon_path.md §6.6, FINDING 2): human-built forum teams, "
                "held out from every training source and SEALED (never read) until X54",
        "drawn_by": "designs/research_state/measurements/forum216b_x54_2026-10-20/draw_forum216b.py",
        "source": {"commit": SOURCE_COMMIT, "data": SOURCE_DATA, "data_sha256": SOURCE_DATA_SHA256,
                   "tier": SOURCE_TIER, "n": SOURCE_TIER_N, "tier_sha256": SOURCE_TIER_SHA256},
        "seed": SEED,
        "quota": QUOTA,
        "curator_cap": CURATOR_CAP,
        "rule": "ids sorted, shuffled once with random.Random(seed), taken in that order; a candidate is skipped if "
                "its ladder_weights_v1 stratum is full or its curator has curator_cap teams, and rejected if the "
                "pinned validator rejects it, a Hidden Power does not play its named type at 70, or it is an exact "
                "team / shares >= 4 mon sets with ONE team of the 719-team pool, home72, away, heldout72, "
                "heldout72_hpfix, forum72, p72 or an already-drawn team",
        "n": len(drawn),
        "content_digest": team_set_digest([d["text"] for d in drawn]),
        "teams": {d["id"]: {"sha256": _sha(files[f"{d['id']}.txt"]), "stratum": d["stratum"], "tier": d["tier"],
                            "curator": d["curator"]} for d in drawn},
        "stats": stats,
        "overlap": overlap_check(drawn),
        "draw_log": log,
    }
    return files, (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    files, man = build()
    tdir, mpath = HERE / f"teams_{NAME}", HERE / f"{NAME}_manifest.json"
    if a.check:
        have = {f.name: f.read_bytes() for f in tdir.glob("*.txt")} if tdir.is_dir() else {}
        same = have == files and mpath.exists() and mpath.read_bytes() == man
        print("reproduces the committed panel" if same else "DOES NOT reproduce the committed panel")
        return 0 if same else 1
    if tdir.exists() and any(tdir.iterdir()):
        raise SystemExit(f"{tdir} exists: a registered panel is never redrawn in place")
    tdir.mkdir(parents=True, exist_ok=True)
    for name, b in files.items():
        (tdir / name).write_bytes(b)
    mpath.write_bytes(man)
    m = json.loads(man)
    print(NAME, m["n"], m["content_digest"], json.dumps(m["stats"]["by_tier"]), json.dumps(m["quota"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
