"""Prototype: every concrete effect line the gen3 sim source can announce, through (a) the Python Gen3Battle +
gen3_effects classification (today's drift check) and (b) the Rust reader (spectator chain + PROBE)."""
import sys
from pathlib import Path

from agents.observation import gen3_effect_sources as S
from main.live.reader import LiveReader, ReaderRefusal

root = Path(sys.argv[1])
PREAMBLE = ["|player|p1|p1user||", "|player|p2|p2user||", "|teamsize|p1|6", "|teamsize|p2|6", "|gametype|singles",
            "|gen|3", "|start", "|switch|p1a: Zappy|Zapdos, L100|100/100", "|switch|p2a: Snorlax|Snorlax, L100, M|100/100",
            "|turn|1", "|move|p2a: Snorlax|Tackle|p1a: Zappy"]
lines = S.concrete_lines(S.scan_emissions(root))
from agents.observation.gen3_effects import GEN3_VOLATILE_TO_SLOT, NOT_A_VOLATILE
r = LiveReader()
agree = disagree = 0
rows = []
for (kw, eff, extra), ems in sorted(lines.items()):
    ids = S.effect_ids_for_line(kw, eff, extra)
    py_bad = sorted(i for i in ids if i not in GEN3_VOLATILE_TO_SLOT and i not in NOT_A_VOLATILE)
    text = "|".join(["", kw, "p2a: Snorlax", eff, *extra])
    rust = None
    try:
        r.open("p1", "p1user", None)
        r.feed(PREAMBLE)
        r.feed([text])
        rust = r.probe()
    except ReaderRefusal as e:
        rust = f"REFUSED {e.kind}: {e}"
        r.close()
        r = LiveReader()
    ok_py, ok_rust = not py_bad, rust is None
    if ok_py == ok_rust:
        agree += 1
    else:
        disagree += 1
    rows.append((ok_py, ok_rust, text, py_bad, rust))
print("lines", len(lines), "agree", agree, "disagree", disagree)
for ok_py, ok_rust, text, py_bad, rust in rows:
    if not (ok_py and ok_rust):
        print(("PY-OK " if ok_py else "PY-BAD") + (" RUST-OK " if ok_rust else " RUST-BAD"), text, py_bad, (rust or "")[:200])
