"""Per-class MARGINALS of the drawn worlds (for §12.2 R-REPLICATE's total-variation check, which a third party
runs against the first implementation's draws). Pooled over states, per checkpoint and arm.

    python -m replica.marginals --rows <rows dir> --out <dir>

Classes: U (never-seen slots: species, item, ability, nature, each move's inclusion), and for SEEN mons only the
facts the searched side had not seen: M (each unrevealed move's inclusion; the type of a bare Hidden Power), I
(item when unrevealed), A (ability when unrevealed), S (nature; EV spread as a string). Each table is a frequency
distribution over the 8 worlds of every included state.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def tid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def unpack(packed: str):
    out = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        out.append({"name": f[0], "species": tid(f[1] or f[0]), "item": f[2], "ability": f[3],
                    "moves": [m for m in f[4].split(",") if m], "nature": f[5], "evs": f[6], "ivs": f[8]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows_dir = Path(os.path.expanduser(a.rows))
    result: dict = {}
    for f in sorted(glob.glob(str(rows_dir / "*" / "*" / "*.jsonl"))):
        cp, arm = Path(f).parts[-3], Path(f).parts[-2]
        if arm == "O":
            continue
        tabs = result.setdefault(cp, {}).setdefault(arm, defaultdict(Counter))
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ok"):
                continue
            seen = r.get("seen", {})
            for w in r["worlds"]:
                for m in unpack(w["team"]):
                    s = seen.get(m["name"])
                    if s is None or s["species"] != m["species"]:
                        tabs["U.species"][m["species"]] += 1
                        tabs["U.item"][m["item"]] += 1
                        tabs["U.ability"][m["ability"]] += 1
                        tabs["U.nature"][m["nature"]] += 1
                        tabs["U.n_sets"]["sets"] += 1
                        for mv in m["moves"]:
                            tabs["U.move"][mv] += 1
                        continue
                    rev = set(s["moves"])
                    tabs["seen.n_sets"]["sets"] += 1
                    for mv in m["moves"]:
                        if mv in rev:
                            continue
                        if mv.startswith("hiddenpower") and "hiddenpower" in rev:
                            tabs["M.bare_hp_type"][mv[len("hiddenpower"):]] += 1
                            continue
                        tabs["M.move"][f"{m['species']}:{mv}"] += 1
                    if s.get("item") is None:
                        tabs["I.item"][f"{m['species']}:{m['item']}"] += 1
                    if s.get("ability") is None:
                        tabs["A.ability"][f"{m['species']}:{m['ability']}"] += 1
                    tabs["S.nature"][f"{m['species']}:{m['nature']}"] += 1
                    tabs["S.spread"][f"{m['species']}:{m['nature']}:{m['evs']}:{m['ivs']}"] += 1
    out = {cp: {arm: {k: dict(v.most_common()) for k, v in t.items()} for arm, t in arms.items()}
           for cp, arms in result.items()}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "marginals.json").write_text(json.dumps(out, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
