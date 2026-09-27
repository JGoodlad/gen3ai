#!/usr/bin/env python3
"""Build ``teams_home72/`` — the ONE team set every round-robin cell of this read draws from.

Source: the 72 team files of ``data/teams/sample`` (the set Foul Play's stock ``home`` directory
was copied from). Three third-party hazards are removed IN THE FILE, so all three clients (our
``main.play``, Foul Play, Metamon's upstream poke-env) read byte-identical, complete pastes:

* **H2 nicknames** — ``Airmure (Skarmory) @ Leftovers`` becomes ``Skarmory @ Leftovers``
  (a ``(M)``/``(F)`` gender tag is kept);
* **H3 Hidden Power IVs** — every Hidden Power move is spelled ``Hidden Power [Type]`` and its
  mon carries the explicit ``IVs:`` line from ``utils.gen3_utils.GEN3_HP_IVS``. The 2026-09-14
  exporter matched only the BRACKETED spelling, so 23 bracketless ``Hidden Power Grass``-style
  moves on 11 of the 72 stock Foul Play files carry no IV line (FINDING in the README);
* **H4 trailing blank lines** — one blank line between mons, one newline at the end.

Every output is validated by the pinned Showdown's own validator (``validate_teams_locally``)
and must hold exactly six mons. Usage (from the tree root, PYTHONPATH=src):

    python make_home72.py <out_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from utils.bridge.team_validator import validate_teams_locally
from utils.gen3_utils import GEN3_HP_IVS
from utils.paths import repo_path

STAT_ORDER = [("hp", "HP"), ("atk", "Atk"), ("def", "Def"), ("spa", "SpA"), ("spd", "SpD"),
              ("spe", "Spe")]
HP_RE = re.compile(r"^-\s*Hidden Power\s*\[?\s*([A-Za-z]+)\s*\]?\s*$", re.IGNORECASE)
NICK_RE = re.compile(r"^(?P<nick>.+?)\s*\((?P<species>[^()]+)\)(?P<rest>\s*(\((?:M|F)\))?\s*(@.*)?)$")


def fix_block(block: str) -> str:
    lines = [ln.rstrip() for ln in block.strip().split("\n")]
    head = lines[0]
    m = NICK_RE.match(head)
    if m and m.group("species").strip() not in ("M", "F"):
        head = f"{m.group('species').strip()}{m.group('rest')}"
    lines[0] = head
    hp_type = None
    for i, ln in enumerate(lines):
        mm = HP_RE.match(ln.strip())
        if mm:
            hp_type = mm.group(1).capitalize()
            lines[i] = f"- Hidden Power [{hp_type}]"
    if hp_type is None:
        return "\n".join(lines)
    lines = [ln for ln in lines if not ln.strip().lower().startswith("ivs:")]
    ivs = GEN3_HP_IVS[hp_type]
    parts = [f"{ivs[k]} {lab}" for k, lab in STAT_ORDER if ivs[k] != 31]
    if parts:
        first_move = next(i for i, ln in enumerate(lines) if ln.startswith("-"))
        lines.insert(first_move, "IVs: " + " / ".join(parts))
    return "\n".join(lines)


def fix_team(text: str) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return "\n\n".join(fix_block(b) for b in blocks) + "\n"


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    src = sorted(repo_path("data", "teams", "sample").glob("*.txt"))
    assert len(src) == 72, len(src)
    texts = {p.name: fix_team(p.read_text()) for p in src}
    for name, text in texts.items():
        blocks = text.strip().split("\n\n")
        assert len(blocks) == 6, (name, len(blocks))
        for b in blocks:
            assert not NICK_RE.match(b.split("\n")[0]) or "(M)" in b or "(F)" in b, (name, b)
            if "Hidden Power" in b and "Hidden Power [Dark]" not in b:
                assert "IVs:" in b, (name, b)
    verdicts = validate_teams_locally("gen3ou", list(texts.values()))
    bad = [(n, v.get("errors")) for n, v in zip(texts, verdicts) if not v.get("valid")]
    if bad:
        print("INVALID:", bad, file=sys.stderr)
        return 1
    for name, text in texts.items():
        (out / name).write_text(text)
    print(f"wrote {len(texts)} validated teams -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
