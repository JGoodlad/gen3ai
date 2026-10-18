#!/usr/bin/env python3
"""Regenerate heldout72 as ``heldout72_hpfix``: the SAME 72 teams by id, only Hidden Power repaired
(``agents.gen3_data.hp_repair``, ``gen3_hp_repair_v1``), and write its manifest.

    python make_heldout72_hpfix.py <hl_05_26/gen3ou dir> <competitive/gen3ou dir> <out measurement dir>

Run from a gen3ai checkout root with PYTHONPATH=src. Deterministic (seed 20261018 + team id per team).

Per team: the SOURCE file named by ``heldout72_manifest.json`` (sha256-checked) -> ``hp_repair.repair_team`` ->
heldout72's own sanitiser (``make_heldout72.fix_team``, with the validator's ``HPivs`` fill table in place of
``GEN3_HP_IVS``; the two differ only on Ghost, which no heldout72 team carried). Then four checks, each a hard
failure: (1) every line outside a repaired set's Hidden Power move line and IVs line is byte-identical to the
heldout72 file; (2) the pinned validator accepts the team (``validate_teams_locally``, as the draw ran it);
(3) ``utils.bridge.hp_set_check``: every Hidden Power set plays the type its line names at power 70 and the
validator leaves its IVs unchanged; (4) the per-team seed and every draw are recorded in the manifest.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from agents.gen3_data import hp_repair as h
from main.anchors.session import load_team_texts, pool_team_texts
from main.anchors.team_overlap import OVERLAP_VERSION, Reference, overlap, parse_items
from utils.bridge.hp_set_check import check_teams
from utils.bridge.team_validator import validate_teams_locally

SEED = 20261018
HERE = Path(__file__).resolve().parent
HL72 = HERE.parents[1] / "heldout_teams_kakuna_2026-10-18"


def _load_make_heldout72():
    spec = importlib.util.spec_from_file_location("make_heldout72", HL72 / "scripts" / "make_heldout72.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    mod.GEN3_HP_IVS = {t: dict(v) for t, v in h.HP_IVS.items()}     # the validator's fill (see docstring)
    return mod


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blocks(text: str) -> list:
    return [b for b in text.strip().split("\n\n")]


def main() -> int:
    src, comp_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mk = _load_make_heldout72()
    man72 = json.loads((HL72 / "heldout72_manifest.json").read_text())
    teams, texts, before, after = {}, {}, collections.Counter(), collections.Counter()
    side_changed = n_rep = 0
    for tid, rec in sorted(man72["teams"].items()):
        raw = (src / rec["source_file"]).read_bytes()
        if sha(raw) != rec["source_sha256"]:
            raise SystemExit(f"source sha256 mismatch: {rec['source_file']}")
        old = (HL72 / "teams_heldout72" / f"{tid}.txt").read_text()
        if sha(old.encode()) != rec["sha256"]:
            raise SystemExit(f"heldout72 sha256 mismatch: {tid}")
        seed = h.team_seed(SEED, tid)
        repaired, reps = h.repair_team(raw.decode(), seed)
        new = mk.fix_team(repaired)
        ob, nb = blocks(old), blocks(new)
        if len(ob) != 6 or len(nb) != 6:
            raise SystemExit(f"{tid}: not six sets")
        rep_slots = {r.slot: r for r in reps}
        for i, (a, b) in enumerate(zip(ob, nb)):
            before[h.hidden_power_type_of(a)] += 1
            after[h.hidden_power_type_of(b)] += 1
            if i not in rep_slots:
                if a != b:
                    raise SystemExit(f"{tid} slot {i}: an unrepaired set changed")
                continue
            r = rep_slots[i]
            if h.hidden_power_type_of(a) != "Dark":
                raise SystemExit(f"{tid} slot {i}: repaired a set heldout72 did not play as HP Dark")
            diff = [(x, y) for x, y in zip(a.split("\n"), b.split("\n")) if x != y]
            al, bl = a.split("\n"), b.split("\n")
            if len(al) != len(bl) or any(not (x.startswith("IVs:") or x.startswith("- Hidden Power"))
                                         for x, _ in diff):
                raise SystemExit(f"{tid} slot {i}: a line other than HP / IVs changed: {diff}")
            if f"- Hidden Power [{r.hp_type}]" not in bl or h.hp_iv_line(r.hp_type) not in bl:
                raise SystemExit(f"{tid} slot {i}: the repaired lines are not the repair's")
            n_rep += 1
            side_changed += h.move_side(r.hp_type) != h.SPECIAL      # HP Dark is special
        texts[tid] = new
        teams[tid] = {"source_file": rec["source_file"], "source_sha256": rec["source_sha256"],
                      "heldout72_sha256": rec["sha256"], "sha256": sha(new.encode()), "team_seed": seed,
                      "repairs": [r.as_dict() for r in reps]}
    order = sorted(texts)
    v1 = validate_teams_locally("gen3ou", [texts[t] for t in order])
    v2 = check_teams([texts[t] for t in order])
    for tid, a, b in zip(order, v1, v2):
        if not a.get("valid") or not b["valid"]:
            raise SystemExit(f"{tid}: validator rejected: {a.get('errors')} {b['errors']}")
        for s in b["sets"]:
            if s["hp_move"] is None:
                continue
            if s["power"] != 70 or s["ivs_in"] != s["ivs_after_validate"] or s["hp_type"] != s["validator_hp_type"]:
                raise SystemExit(f"{tid}: {s['species']} HP check failed: {s}")
        for r in teams[tid]["repairs"]:
            s = b["sets"][r["slot"]]
            if s["hp_type"] != r["hp_type"] or s["ivs_in"] != r["ivs"]:
                raise SystemExit(f"{tid}: slot {r['slot']} plays {s['hp_type']}, wrote {r['hp_type']}")
    tdir = out / "teams_heldout72_hpfix"
    tdir.mkdir(parents=True, exist_ok=True)
    for tid in order:
        (tdir / f"{tid}.txt").write_bytes(texts[tid].encode())
    pool = Reference(parse_items([(f"pool{i:03d}", t) for i, t in enumerate(pool_team_texts())]))
    comp = Reference(parse_items(load_team_texts(comp_dir)))
    items = parse_items([(t, texts[t]) for t in order])
    reps = [r for t in order for r in teams[t]["repairs"]]
    manifest = {
        "schema": "heldout72_hpfix_manifest_v1",
        "repair": h.REPAIR_VERSION, "seed": SEED,
        "seed_rule": "team_seed = int(sha256(f'{repair}|{seed}|{team_id}')[:8 bytes], big-endian)",
        "derived_from": "heldout_teams_kakuna_2026-10-18/heldout72_manifest.json", "source_dir": str(src),
        "n": len(order), "matcher": OVERLAP_VERSION, "teams": teams,
        "summary": {
            "hp_sets": sum(before.values()) - before[None], "repaired": n_rep,
            "hp_type_before": dict(sorted((k, v) for k, v in before.items() if k)),
            "hp_type_after": dict(sorted((k, v) for k, v in after.items() if k)),
            "repaired_side_changed": side_changed,
            "side_decided_by": dict(collections.Counter(r["decided_by"] for r in reps)),
            "side": dict(collections.Counter(r["side"] for r in reps)),
            "prior_source": dict(collections.Counter(r["prior_source"] for r in reps)),
            "validator": "validate_teams_locally gen3ou + utils.bridge.hp_set_check: 72 / 72 valid; every HP set "
                         "power 70, IVs unchanged by the validator, played type == written type",
        },
        "overlap_check": {"training_pool": overlap(items, pool), "competitive": overlap(items, comp)},
    }
    (out / "heldout72_hpfix_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
