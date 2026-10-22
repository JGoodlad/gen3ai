"""X52's two halves -- the PRACTISED 36 and the UNPRACTISED 36 (``design_beat_metamon_path.md`` §6.2; the X35 launch
sheet §0 AM-X52). It plays nothing.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src nice -n 19 taskset -c 5-7,13-15 python <this> [--check]

The source is ``w_forum_v1``'s tier A AS COMMITTED AT THE PRODUCTION PIN (``SOURCE_*``; the corpus already dropped every
registered panel -- forum216b included -- with its near neighbours, and the pool's exact teams), read from git so the
draw reproduces whatever main later does to the corpus. 72 teams are drawn with forum72's stratum quotas and curator
cap, rejecting any team that is an exact team of, or shares >= 4 mon sets with one team of, the 719-team pool, a
registered panel, or an already-drawn team (so the halves are NOT near neighbours of each other, and every team is as
unfamiliar to P* as the U panels). The 72 are then split 36 / 36 inside each stratum by ``SPLIT_SEED``.

Writes ``teams_practised/<id>.txt``, ``teams_unpractised/<id>.txt`` and ``x52_teams_manifest.json`` beside this script;
``--check`` exits 1 unless the committed files reproduce byte for byte. The practised files are what X52's
``--trainee-teams`` names (``main.x35.x52``); nothing else may train on either half.
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
SCHEMA = "x52_practice_halves_v1"
SOURCE_COMMIT = "f4b25238def57900c0305bb2a3a5a8f9edaef285"   # X35's PRODUCTION pin (x35_pin_batch1_production.txt)
SOURCE_DATA = "src/utils/team_corpora/w_forum_v1/teams.jsonl.gz"
SOURCE_DATA_SHA256 = "7f42d95f7f08f667738fe6e8f2af33ede93f5dec78af6fa48ed3f72f3d231336"
SOURCE_TIER = "A"
SOURCE_TIER_N = 169
SOURCE_TIER_SHA256 = "3ac32aa67e1d196bb4f764fc835501126e266836d08851ef5be39d71fd44aeb4"
SEED = 5200
SPLIT_SEED = 5201
#: forum72's stratum counts (CORE 4 / NEAR 54 / FAR 14 of 72) and its curator cap (14 per 72), except that tier A can
#: fill only 3 CORE teams under the rejection rules (6 in supply), so CORE is 2 and NEAR 56: every quota is EVEN, so
#: each stratum splits exactly in half.
QUOTA = {"CORE": 2, "NEAR": 56, "FAR": 14}
N = sum(QUOTA.values())
CURATOR_CAP = 14
REF_PANELS = ("home72", "away", "heldout72", "heldout72_hpfix", "forum72", "p72")
STRATUM_REF = ("home72", "away")
HALVES = ("practised", "unpractised")


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
    tier = [r for r in rows if r["tier"] == SOURCE_TIER]
    if len(tier) != SOURCE_TIER_N or rows[:SOURCE_TIER_N] != tier:
        raise SystemExit(f"tier {SOURCE_TIER} is not the first {SOURCE_TIER_N} rows")
    h = hashlib.sha256()
    for r in tier:
        h.update(r["text"].encode())
        h.update(b"\x00")
    if h.hexdigest() != SOURCE_TIER_SHA256:
        raise SystemExit(f"tier {SOURCE_TIER} sha256 {h.hexdigest()} != {SOURCE_TIER_SHA256}")
    return tier


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
        raise SystemExit(f"the draw could not fill its quotas: {dict(taken)} != {QUOTA} (supply {dict(supply)})")
    stats = {"supply_by_stratum": dict(supply), "per_curator": dict(per_cur.most_common())}
    return drawn, log, stats


def split(drawn: List[Dict[str, Any]]) -> Dict[str, str]:
    """``{id: half}``: inside each stratum, ids sorted, shuffled once with ``SPLIT_SEED``, the first half practised."""
    rng = random.Random(SPLIT_SEED)
    out: Dict[str, str] = {}
    for s in QUOTA:
        ids = sorted(d["id"] for d in drawn if d["stratum"] == s)
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            out[i] = HALVES[0] if j < len(ids) // 2 else HALVES[1]
    return out


def overlap_check(drawn: List[Dict[str, Any]], half: Dict[str, str]) -> Dict[str, Any]:
    """The max shared mon sets of every drawn team with each reference, and ACROSS the halves (the leak X52's transfer
    read must not have: an unpractised team that near-copies a practised one)."""
    refs = references()
    out = {}
    for name, ref in refs.items():
        hist: collections.Counter = collections.Counter()
        for d in drawn:
            hist[ref.max_shared(TO.parse_team(d["text"]))[0]] += 1
        out[name] = {str(k): hist.get(k, 0) for k in range(7)}
    pr = TO.Reference([TO.parse_team(d["text"], d["id"]) for d in drawn if half[d["id"]] == HALVES[0]])
    cross: collections.Counter = collections.Counter()
    cross_species: collections.Counter = collections.Counter()
    pr_species = [TO.parse_team(d["text"]).species6 for d in drawn if half[d["id"]] == HALVES[0]]
    for d in drawn:
        if half[d["id"]] == HALVES[1]:
            t = TO.parse_team(d["text"])
            cross[pr.max_shared(t)[0]] += 1
            cross_species[max(len(t.species6 & s) for s in pr_species)] += 1
    return {"max_shared_mon_sets_hist": out,
            "unpractised_vs_practised_max_shared_mon_sets_hist": {str(k): cross.get(k, 0) for k in range(7)},
            "unpractised_vs_practised_max_shared_species_hist": {str(k): cross_species.get(k, 0) for k in range(7)}}


def build() -> Tuple[Dict[str, Dict[str, bytes]], bytes]:
    from main.anchors.session import team_set_digest

    drawn, log, stats = draw()
    half = split(drawn)
    files = {h: {f"{d['id']}.txt": (d["text"] + "\n").encode() for d in drawn if half[d["id"]] == h} for h in HALVES}
    manifest = {
        "schema": SCHEMA,
        "what": "X52's practice halves (design_beat_metamon_path.md §6.2): the practised 36 are X52's --trainee-teams; "
                "the unpractised 36 are its within-population transfer test. Both are read against Kakuna",
        "drawn_by": "designs/research_state/measurements/x52_practice_ceiling_2026-10-22/draw_x52_teams.py",
        "source": {"commit": SOURCE_COMMIT, "data": SOURCE_DATA, "data_sha256": SOURCE_DATA_SHA256,
                   "tier": SOURCE_TIER, "n": SOURCE_TIER_N, "tier_sha256": SOURCE_TIER_SHA256},
        "seed": SEED, "split_seed": SPLIT_SEED, "quota": QUOTA, "curator_cap": CURATOR_CAP,
        "rule": "ids sorted, shuffled once with random.Random(seed), taken in that order; a candidate is skipped if its "
                "ladder_weights_v1 stratum is full or its curator has curator_cap teams, and rejected if the pinned "
                "validator rejects it, a Hidden Power does not play its named type at 70, or it is an exact team / "
                "shares >= 4 mon sets with ONE team of the 719-team pool, home72, away, heldout72, heldout72_hpfix, "
                "forum72, p72 or an already-drawn team. Split: inside each stratum, ids sorted, shuffled once with "
                "random.Random(split_seed) (one generator, strata in quota order), the first half practised",
        "n": len(drawn),
        "halves": {h: {"n": len(files[h]),
                       "content_digest": team_set_digest([d["text"] for d in drawn if half[d["id"]] == h]),
                       "by_stratum": dict(collections.Counter(d["stratum"] for d in drawn if half[d["id"]] == h))}
                   for h in HALVES},
        "teams": {d["id"]: {"half": half[d["id"]], "sha256": _sha(files[half[d["id"]]][f"{d['id']}.txt"]),
                            "stratum": d["stratum"], "curator": d["curator"]} for d in drawn},
        "stats": stats,
        "overlap": overlap_check(drawn, half),
        "draw_log": log,
    }
    return files, (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    files, man = build()
    mpath = HERE / "x52_teams_manifest.json"
    dirs = {h: HERE / f"teams_{h}" for h in HALVES}
    if a.check:
        same = mpath.exists() and mpath.read_bytes() == man and all(
            ({f.name: f.read_bytes() for f in dirs[h].glob("*.txt")} if dirs[h].is_dir() else {}) == files[h]
            for h in HALVES)
        print("reproduces the committed halves" if same else "DOES NOT reproduce the committed halves")
        return 0 if same else 1
    if any(d.exists() and any(d.iterdir()) for d in dirs.values()):
        raise SystemExit("a half exists: registered halves are never redrawn in place")
    for h in HALVES:
        dirs[h].mkdir(parents=True, exist_ok=True)
        for name, b in files[h].items():
            (dirs[h] / name).write_bytes(b)
    mpath.write_bytes(man)
    m = json.loads(man)
    print(m["n"], json.dumps(m["halves"]), json.dumps(m["overlap"]["unpractised_vs_practised_max_shared_species_hist"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
