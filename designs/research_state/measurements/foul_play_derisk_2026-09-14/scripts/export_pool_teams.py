"""Export gen3ai pool teams as Showdown pastes with EXPLICIT Hidden Power IVs.

Why the IV line has to be written out: our pool `.txt` files omit it, and
`utils.gen3_utils.fix_gen3_hp_ivs` patches it in at pack time inside
`Gen3Teambuilder`. Foul Play has no such step — its `fp/teams/team_converter.py`
packs whatever the paste says, so a `Hidden Power [Grass]` with no IV line packs
as 31/31/31/31/31/31 (= Hidden Power Dark) and the Showdown validator refuses the
team. Writing the IVs into the paste gives BOTH clients the same legal team.

Usage:
    python export_pool_teams.py <src_dir> <dst_dir>
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "dev", "gen3ai", "src"))
from utils.gen3_utils import GEN3_HP_IVS  # noqa: E402

STAT_ORDER = [("hp", "HP"), ("atk", "Atk"), ("def", "Def"), ("spa", "SpA"),
              ("spd", "SpD"), ("spe", "Spe")]
HP_RE = re.compile(r"^-\s*Hidden Power\s*\[([A-Za-z]+)\]", re.IGNORECASE)


def iv_line(hp_type: str) -> str:
    ivs = GEN3_HP_IVS[hp_type.capitalize()]
    parts = [f"{ivs[k]} {label}" for k, label in STAT_ORDER if ivs[k] != 31]
    return "IVs: " + " / ".join(parts) if parts else ""


def fix_block(block: str) -> str:
    lines = [ln.rstrip() for ln in block.strip().split("\n")]
    hp_type = None
    for ln in lines:
        m = HP_RE.match(ln.strip())
        if m:
            hp_type = m.group(1)
            break
    # drop any existing IVs line; we re-derive it
    lines = [ln for ln in lines if not ln.strip().lower().startswith("ivs:")]
    if hp_type is None:
        return "\n".join(lines)
    new = iv_line(hp_type)
    if not new:
        return "\n".join(lines)
    # insert before the first move line
    for i, ln in enumerate(lines):
        if ln.strip().startswith("-"):
            return "\n".join(lines[:i] + [new] + lines[i:])
    return "\n".join(lines + [new])


def fix_team(text: str) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return "\n\n".join(fix_block(b) for b in blocks) + "\n"


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(src, name)) as f:
            text = f.read()
        with open(os.path.join(dst, name), "w") as f:
            f.write(fix_team(text))
        n += 1
    print(f"exported {n} teams -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
