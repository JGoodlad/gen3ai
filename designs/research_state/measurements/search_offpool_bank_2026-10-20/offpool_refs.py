"""THE OFF-POOL PROOF and THE OFF-POOL DECOY CORPUS (``PREREGISTRATION.md`` §2 and §6). Import after
``offpool_common.use_code`` (it needs the checkout's ``main`` / ``agents`` / ``utils``).

    python -I offpool_refs.py --code <8fe4fb89 checkout> proof     -> offpool_proof.json (this folder)
    python -I offpool_refs.py --code <8fe4fb89 checkout> corpus    -> <archive>/decoy_corpus.json

THE REFERENCE (every team X26's lineage trained against). X26 (``rb_x26_s2601``) is a FRESH run (``lineage.role`` fresh,
no ancestry, no teacher), trained start to end at its pin ``9c033489`` (``pin_history``), its trainee and its
opponents (self-play snapshots and the scripted bots alike) drawing from the team pool ``data/teams``
(``matchup_history``: trainee ``default_biased`` over the pool, opponents ``pool``; no ``team_mix``). X26's pin predates
``gen3_pinned_data_root_v1``, so it read MAIN's ``data/teams``; no commit after the pin touches ``data/teams``
(``git log 9c033489..HEAD -- data/teams`` is empty, checked here), so that tree IS the pin's. The reference is:
  * R_PIN: EVERY team file under ``data/teams`` at the pin (``git archive``; a superset of the 719 the loader keeps);
  * R_PACKED: stage 1's ``pool_pin.json``, the pool exactly as the pin's trainer packed it (sha-checked);
  * R_COMP: Metamon's ``competitive/gen3ou`` (20 teams; 19 are already pool copies) — not trained against, kept as the
    heldout72 draw kept it.
THE CHECKS (``gen3_team_overlap_v1``: a mon set = species + sorted moves, Hidden Power typed): no read team equals a
reference team (loose or strict), none shares >= 4 mon sets with any ONE reference team, and the packed texts are
disjoint from R_PACKED by sha256. A violation is a hard stop.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offpool_common as E  # noqa: E402

HP_SEED = 20261018          # heldout72_hpfix's repair seed (its manifest's `seed`)


def pin_team_texts() -> List[Tuple[str, str]]:
    """``(relative path, text)`` of every file under ``data/teams`` at X26's pin that parses as >= 1 Pokemon."""
    from main.anchors.team_overlap import parse_team

    later = subprocess.run(["git", "-C", str(E.MAIN), "log", "--format=%H", f"{E.X26_PIN}..origin/main", "--",
                            "data/teams"], capture_output=True, text=True, check=True).stdout.split()
    if later:
        raise SystemExit(f"commits after X26's pin touch data/teams: {later[:3]}")
    out = []
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "t.tar"
        subprocess.run(["git", "-C", str(E.MAIN), "archive", "-o", str(tar), E.X26_PIN, "data/teams"], check=True)
        with tarfile.open(tar) as tf:
            for m in sorted(tf.getmembers(), key=lambda m: m.name):
                if not m.isfile():
                    continue
                text = tf.extractfile(m).read().decode("utf-8", errors="replace")
                if parse_team(text).mons:
                    out.append((m.name, text))
    return out


def pool_pin_packed() -> List[str]:
    blob = json.loads(E.POOL_PIN.read_text())
    if hashlib.sha256("\n".join(blob["teams"]).encode()).hexdigest() != blob["sha256"]:
        raise SystemExit("pool_pin.json does not match its sha256")
    return list(blob["teams"])


def references():
    from main.anchors.session import load_team_texts
    from main.anchors.team_overlap import Reference, parse_items

    pin = pin_team_texts()
    comp = load_team_texts(E.COMPETITIVE_DIR)
    return {"R_PIN": (pin, Reference(parse_items(pin))), "R_COMP": (comp, Reference(parse_items(comp)))}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def proof() -> dict:
    """The off-pool proof of the 72 opponent teams (module docstring). Hard stop on any violation."""
    from main.anchors.team_overlap import OVERLAP_VERSION, overlap, parse_items
    from utils.teambuilder import Gen3Teambuilder

    texts = E.heldout_texts()
    order = sorted(texts)
    read = parse_items([(t, texts[t]) for t in order])
    if any(len(t.mons) != 6 for t in read):
        raise SystemExit("a heldout team does not parse as six Pokemon")
    refs = references()
    out = {"matcher": OVERLAP_VERSION, "read": "heldout72_hpfix (72 teams, hl_05_26 by id, Hidden Power repaired)",
           "references": {}}
    for name, (items, ref) in refs.items():
        ov = overlap(read, ref)
        out["references"][name] = {"n_files": len(items), **ov}
        if ov["exact_team"] or ov["exact_team_strict"] or ov["teams_sharing_ge4"]:
            raise SystemExit(f"OFF-POOL PROOF FAILED against {name}: {ov}")
    packed = list(Gen3Teambuilder([texts[t] for t in order]).packed_teams)
    pin_packed = pool_pin_packed()
    mons_pin = {e for t in pin_packed for e in t.split("]")}
    shared_entries = [sum(1 for e in p.split("]") if e in mons_pin) for p in packed]
    inter = {_sha(p) for p in packed} & {_sha(p) for p in pin_packed}
    out["references"]["R_PACKED"] = {
        "n_teams": len(pin_packed), "distinct": len(set(pin_packed)), "sha256": json.loads(E.POOL_PIN.read_text())["sha256"],
        "packed_team_sha_intersection": len(inter),
        "max_identical_packed_mon_entries_per_team": max(shared_entries),
        "identical_packed_mon_entries_hist": {str(k): shared_entries.count(k) for k in range(7)}}
    if inter or max(shared_entries) >= 4:
        raise SystemExit(f"OFF-POOL PROOF FAILED against R_PACKED: {out['references']['R_PACKED']}")
    out["heldout_packed_sha256"] = {t: _sha(p) for t, p in zip(order, packed)}
    out["verdict"] = ("OFF-POOL: 0 exact teams (loose and strict), 0 teams sharing >= 4 mon sets with any one "
                      "reference team, 0 packed teams in common with the pin's packed pool")
    return out


def _make_heldout72():
    """heldout72's own sanitiser with the validator's Hidden Power IV table (``make_heldout72_hpfix.py``'s loader)."""
    from agents.gen3_data import hp_repair as h

    p = E.MEAS / "heldout_teams_kakuna_2026-10-18" / "scripts" / "make_heldout72.py"
    spec = importlib.util.spec_from_file_location("make_heldout72", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    mod.GEN3_HP_IVS = {t: dict(v) for t, v in h.HP_IVS.items()}
    return mod


def corpus() -> dict:
    """The OFF-POOL DECOY CORPUS: every ``hl_05_26`` team the engine plays (``utils.ladder_corpus``'s kept files), the 72
    heldout teams removed, Hidden Power repaired exactly as heldout72_hpfix was (``hp_repair``, seed 20261018 + team id,
    heldout72's sanitiser), then DROPPED if it equals or shares >= 4 mon sets with any one R_PIN or R_COMP team, or
    its packed text is in R_PACKED; validated and packed by the trainer's own ``Gen3Teambuilder``; exact packed
    duplicates kept once."""
    from agents.gen3_data import hp_repair as h
    from main.anchors.team_overlap import parse_team
    from utils import ladder_corpus as LC
    from utils.teambuilder import Gen3Teambuilder

    mk = _make_heldout72()
    kept_files = {r["file"] for r in LC._all_rows()}
    held_files = {rec["source_file"] for rec in json.loads(E.HELDOUT_MANIFEST.read_text())["teams"].values()}
    refs = references()
    pin_packed = {_sha(p) for p in pool_pin_packed()}
    counts = {"source_files": 0, "engine_unplayable": 0, "heldout": 0, "not_six": 0, "pool_exact": 0,
              "pool_near_ge4": 0, "validator_drop": 0, "packed_in_pool": 0, "packed_duplicate": 0}
    texts, ids = [], []
    for p in sorted(E.HL_SOURCE.glob("team_*.gen3ou_team")):
        counts["source_files"] += 1
        if p.name not in kept_files:
            counts["engine_unplayable"] += 1
            continue
        if p.name in held_files:
            counts["heldout"] += 1
            continue
        tid = p.name.split(".")[0]
        rep, _ = h.repair_team(p.read_text(), h.team_seed(HP_SEED, tid))
        text = mk.fix_team(rep)
        t = parse_team(text, tid)
        if len(t.mons) != 6:
            counts["not_six"] += 1
            continue
        if any(ref.keys[t.key] or ref.strict_keys[t.strict_key] for _, ref in refs.values()):
            counts["pool_exact"] += 1
            continue
        if any(ref.max_shared(t)[0] >= 4 for _, ref in refs.values()):
            counts["pool_near_ge4"] += 1
            continue
        texts.append(text)
        ids.append(tid)
    tb = Gen3Teambuilder(texts)
    key_to_id = {hashlib.sha1(x.strip().encode()).hexdigest()[:10]: i for x, i in zip(texts, ids)}
    counts["validator_drop"] = len(texts) - len(tb.packed_teams)
    teams, tids, seen = [], [], set()
    for packed, k in zip(tb.packed_teams, tb._pool_keys):
        if _sha(packed) in pin_packed:
            counts["packed_in_pool"] += 1
            continue
        if packed in seen:
            counts["packed_duplicate"] += 1
            continue
        seen.add(packed)
        teams.append(packed)
        tids.append(key_to_id[k])
    blob = {"schema": "search_offpool_decoy_corpus_v1", "code": E.CODE_SHA, "hp_repair": h.REPAIR_VERSION,
            "hp_seed": HP_SEED, "ladder_corpus_sha256": LC.manifest()["data_sha256"], "counts": counts,
            "n": len(teams), "ids": tids, "teams": teams, "sha256": hashlib.sha256("\n".join(teams).encode()).hexdigest()}
    return blob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("what", choices=("proof", "corpus"))
    a = ap.parse_args()
    E.use_code(Path(a.code))
    os.chdir(str(Path(a.code).resolve()))                # the validator bridge resolves deps/ from the checkout
    if a.what == "proof":
        out = proof()
        (E.HERE / "offpool_proof.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(out["verdict"])
    else:
        blob = corpus()
        p = E.ARCH / "decoy_corpus.json"
        if p.exists():
            raise SystemExit(f"{p} exists: the corpus is built once")
        E.ARCH.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(blob) + "\n")
        print(json.dumps({k: blob[k] for k in ("counts", "n", "sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
