#!/usr/bin/env python3
"""Draw ``teams_heldout72/`` by the README's draw rule (written before the draw) and write its manifest.

    python make_heldout72.py <hl_05_26/gen3ou dir> <competitive/gen3ou dir> <out measurement dir>

Run from a gen3ai checkout root with PYTHONPATH=src (the pool resolves against that checkout's data/teams;
the validator runs that checkout's pinned Showdown). Deterministic: the same inputs give the same 72 files.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

from main.anchors.session import load_team_texts, pool_team_texts
from main.anchors.team_overlap import OVERLAP_VERSION, Reference, hidden_power_type, overlap, parse_items, parse_team
from utils.bridge.team_validator import validate_teams_locally
from utils.gen3_utils import GEN3_HP_IVS

SEED = 20261018
N = 72
MAX_SHARED = 3          # rule 2: >= 4 of 6 mon sets in common with one reference team rejects
BATCH = 40              # candidates validated per validator call
STAT_ORDER = [("hp", "HP"), ("atk", "Atk"), ("def", "Def"), ("spa", "SpA"), ("spd", "SpD"), ("spe", "Spe")]
HP_RE = re.compile(r"^-\s*Hidden Power\s*\[?\s*([A-Za-z]*)\s*\]?\s*$", re.IGNORECASE)
NICK_RE = re.compile(r"^(?P<nick>.+?)\s*\((?P<species>[^()]+)\)(?P<rest>\s*(\((?:M|F)\))?\s*(@.*)?)$")
IV_RE = re.compile(r"(\d+)\s+(HP|Atk|Def|SpA|SpD|Spe)", re.IGNORECASE)


def fix_block(block: str) -> str:
    lines = [ln.rstrip() for ln in block.strip().split("\n")]
    m = NICK_RE.match(lines[0])
    if m and m.group("species").strip() not in ("M", "F"):
        lines[0] = f"{m.group('species').strip()}{m.group('rest')}"
    ivs = {}
    for ln in lines:
        if ln.strip().lower().startswith("ivs:"):
            ivs = {k.lower(): int(v) for v, k in IV_RE.findall(ln)}
    hp_type = None
    for i, ln in enumerate(lines):
        mm = HP_RE.match(ln.strip())
        if mm:
            hp_type = (mm.group(1) or hidden_power_type(ivs)).capitalize()
            lines[i] = f"- Hidden Power [{hp_type}]"
    if hp_type is None:
        return "\n".join(lines)
    lines = [ln for ln in lines if not ln.strip().lower().startswith("ivs:")]
    parts = [f"{GEN3_HP_IVS[hp_type][k]} {lab}" for k, lab in STAT_ORDER]
    first_move = next(i for i, ln in enumerate(lines) if ln.startswith("-"))
    lines.insert(first_move, "IVs: " + " / ".join(parts))
    return "\n".join(lines)


def fix_team(text: str) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return "\n\n".join(fix_block(b) for b in blocks) + "\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    src, comp_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    names = sorted(p.name for p in src.glob("*.gen3ou_team"))
    random.Random(SEED).shuffle(names)
    pool = Reference(parse_items([(f"pool{i:03d}", t) for i, t in enumerate(pool_team_texts())]))
    comp = Reference(parse_items(load_team_texts(comp_dir)))
    drawn: dict = {}
    keys = set()
    log = []
    i = 0
    while len(drawn) < N and i < len(names):
        batch = []
        while len(batch) < BATCH and i < len(names):
            name = names[i]
            i += 1
            text = fix_team((src / name).read_text())
            team = parse_team(text, name)
            reason = None
            if len(team.mons) != 6 or len(text.strip().split("\n\n")) != 6:
                reason = "not_six_mons"
            elif team.key in keys:
                reason = "exact_dup_of_drawn"
            else:
                for rname, ref in (("pool", pool), ("competitive", comp)):
                    if ref.keys[team.key]:
                        reason = f"exact_{rname}"
                        break
                    shared, other = ref.max_shared(team)
                    if shared > MAX_SHARED:
                        reason = f"shared{shared}_{rname}:{other}"
                        break
            if reason:
                log.append({"file": name, "rejected": reason})
            else:
                batch.append((name, text, team))
        if not batch:
            continue
        verdicts = validate_teams_locally("gen3ou", [t for _, t, _ in batch])
        for (name, text, team), v in zip(batch, verdicts):
            if len(drawn) >= N:
                log.append({"file": name, "rejected": "not_needed_after_72"})
            elif not v.get("valid"):
                log.append({"file": name, "rejected": "validator", "errors": v.get("errors")})
            elif team.key in keys:
                log.append({"file": name, "rejected": "exact_dup_of_drawn"})
            else:
                drawn[name] = text
                keys.add(team.key)
                log.append({"file": name, "accepted": len(drawn)})
    if len(drawn) < N:
        print(f"only {len(drawn)} teams passed", file=sys.stderr)
        return 1
    tdir = out / "teams_heldout72"
    tdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, text in drawn.items():
        stem = name.rsplit(".", 1)[0]
        data = text.encode()
        (tdir / f"{stem}.txt").write_bytes(data)
        files[stem] = {"source_file": name, "source_sha256": sha((src / name).read_bytes()),
                       "sha256": sha(data)}
    teams = parse_items([(k, drawn[v["source_file"]]) for k, v in files.items()])
    manifest = {
        "schema": "heldout72_manifest_v1",
        "source_dir": str(src), "source_n_files": len(names), "seed": SEED, "n": N,
        "max_shared_mon_sets_allowed": MAX_SHARED, "matcher": OVERLAP_VERSION,
        "candidates_examined": i, "teams": files,
        "overlap_check": {"training_pool": overlap(teams, pool), "competitive": overlap(teams, comp)},
    }
    (out / "heldout72_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (out / "draw_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    print(f"drew {N} of {i} candidates examined -> {tdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
