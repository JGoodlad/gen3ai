#!/usr/bin/env python3
"""gen_dex_golden.py — dex parity harness (the source-of-truth differential).

Dumps the `agents.gen3_data` facade's view of every species / move (incl. the
DERIVED category) / type-chart cell / nature / learnset to a flat golden file.
The Rust dex integration test (`tests/dex_parity.rs`) then asserts its own load
reproduces every value — so the Rust dex and the Python runtime agree by
construction, and the one piece of *logic* (Gen-3 move-category derivation) is
pinned, not just the raw JSON.

Run (needs the project conda env):

    /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \\
        src/rust_sim/harness/gen_dex_golden.py
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Output: ../tests/vectors/dex_golden.txt
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agents import gen3_data as g3

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data" / "pokemon"          # <repo>/data/pokemon
OUT = HERE.parent / "tests" / "vectors" / "dex_golden.txt"


def num(x: float) -> str:
    """Compact, parse-stable float formatting (matches a Rust f64 parse)."""
    return f"{float(x):.6g}"


def main() -> None:
    species_raw = json.loads((DATA / "gen3_species.json").read_text())
    moves_raw = json.loads((DATA / "gen3_moves.json").read_text())
    learn_raw = json.loads((DATA / "gen3_learnset.json").read_text())

    lines: list[str] = []

    # SPECIES <id> <num> <hp> <atk> <def> <spa> <spd> <spe> <type1[,type2]>
    for sid in sorted(species_raw):
        sd = g3.species.get(sid)
        bs = sd.base_stats
        types = ",".join(sd.types) if sd.types else "-"
        lines.append(
            f"SPECIES\t{sid}\t{sd.num}\t{bs['hp']}\t{bs['atk']}\t{bs['def']}\t"
            f"{bs['spa']}\t{bs['spd']}\t{bs['spe']}\t{types}"
        )

    # MOVE <id> <num> <bp> <typename> <category> <accuracy>
    for mid in sorted(moves_raw):
        md = g3.moves.get(mid)
        lines.append(
            f"MOVE\t{mid}\t{md.num}\t{md.base_power}\t{md.type.name}\t"
            f"{md.category.name}\t{md.accuracy}"
        )

    # TYPE <defending> <attacking> <multiplier>
    chart = g3.type_chart.chart()
    for def_t in sorted(chart):
        for att_t in sorted(chart[def_t]):
            lines.append(f"TYPE\t{def_t}\t{att_t}\t{num(chart[def_t][att_t])}")

    # NATURE <name> <atk> <def> <spa> <spd> <spe>
    nat = g3.natures.multipliers()
    for name in sorted(nat):
        m = nat[name]
        lines.append(
            f"NATURE\t{name}\t{num(m['atk'])}\t{num(m['def'])}\t"
            f"{num(m['spa'])}\t{num(m['spd'])}\t{num(m['spe'])}"
        )

    # LEARNSET <species> <sorted,csv,of,move,ids>
    for sid in sorted(learn_raw):
        legal = g3.learnset.get_legal_moves(sid) or frozenset()
        lines.append(f"LEARNSET\t{sid}\t{','.join(sorted(legal))}")

    header = [
        "# Dex parity golden, dumped from agents.gen3_data (the runtime source of truth).",
        "# Regenerate: PYTHONPATH=src python3 src/rust_sim/harness/gen_dex_golden.py",
        "# The Rust dex (tests/dex_parity.rs) must reproduce every line.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + lines) + "\n")
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln.split("\t", 1)[0]] = counts.get(ln.split("\t", 1)[0], 0) + 1
    print(f"wrote {len(lines)} lines -> {os.path.relpath(OUT)}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
