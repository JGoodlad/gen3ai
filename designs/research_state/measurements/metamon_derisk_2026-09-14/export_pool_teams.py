#!/usr/bin/env python3
"""Export OUR gen3ou team pool as Metamon `TeamSet` files.

Metamon's ``TeamSet`` samples a directory of Showdown-export files named
``*.<battle_format>_team``. Our pool lives in ``data/teams`` behind
``utils.team_loader.TeamLoader`` and reaches a battle through ``utils.teambuilder
.Gen3Teambuilder``, which does two things a raw file does NOT: it drops teams the local
validator rejects, and it applies ``fix_gen3_hp_ivs`` (Hidden Power's Gen-3 IV spread is implied
by the move's type, and a paste that leaves the IVs at 31 gets the WRONG HP type on the server).

So the export re-does both, and the IV fix is applied as a TEXT edit on the original paste — an
``IVs:`` line inserted or replaced per Pokémon — rather than by re-rendering a parsed team, so
everything else in the file is the paste we already train against, byte for byte.

Run from the MAIN checkout (it owns ``data/``):

    export PYTHONPATH=$PYTHONPATH:src
    python3 <this file> --out $METAMON_CACHE_DIR/teams/gen3ai_pool/gen3ou
"""

import argparse
import hashlib
import json
import os
import re
import sys

from poke_env.teambuilder import Teambuilder

from utils.gen3_utils import GEN3_HP_IVS
from utils.team_loader import TeamLoader

_STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]

#: ``Nickname (Species)`` — the species in parentheses. ``(M)``/``(F)`` is a GENDER marker, not a
#: species, and must not be mistaken for one.
_NICKNAME_LINE = re.compile(r"^(?P<nick>[^()]+?)\s*\((?P<species>[^()]+)\)(?P<rest>.*)$")


def strip_nickname(line: str) -> str:
    """``Airmure (Skarmory) @ Leftovers`` -> ``Skarmory @ Leftovers``.

    🚨 WHY THIS IS NOT COSMETIC. Metamon runs UPSTREAM poke-env (0.8.3.3); we run our vendored
    fork. On a nickname line with NO ITEM (``Airmure (Skarmory)``) the two parsers DISAGREE:
    upstream leaves the species field EMPTY and packs the entire string as the nickname, which
    Showdown normalises to ``airmureskarmory`` and rejects with
    ``The Pokemon "airmureskarmory" does not exist`` — measured 2026-09-14, and it stalls the
    match rather than failing it. Our fork parses the same line correctly. A nickname carries no
    battle meaning, so dropping it removes the dependency on whose parser reads the file.
    """
    m = _NICKNAME_LINE.match(line.rstrip())
    if not m:
        return line
    species = m.group("species").strip()
    if species.upper() in {"M", "F"}:          # a gender marker, not a nickname
        return line
    return f"{species}{m.group('rest')}".rstrip()


class _Parser(Teambuilder):
    """Only here for ``parse_showdown_team``; never yields."""

    def yield_team(self):  # pragma: no cover - never called
        raise NotImplementedError


def _hp_type(move: str):
    match = re.search(r"\[(\w+)\]", move)
    if match:
        return match.group(1).capitalize()
    return move.lower().replace(" ", "").replace("hiddenpower", "").capitalize()


def _iv_line(hp_type: str) -> str:
    """The non-31 IVs for `hp_type`, in Showdown's export order and syntax."""
    ivs = GEN3_HP_IVS[hp_type]
    key = {"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}
    parts = [f"{v} {s}" for s in _STAT_ORDER for v in [ivs.get(key[s], 31)] if v != 31]
    return "IVs: " + " / ".join(parts) if parts else ""


#: Mutable counter so `fix_team_text` can report how many nickname lines it rewrote.
nicknames_stripped = [0]


def fix_team_text(text: str, parser: _Parser) -> str:
    """Return `text` with each Hidden-Power mon's ``IVs:`` line set to the Gen-3 spread.

    Only mons whose IVs are absent or all-31 are touched — the same condition
    ``fix_gen3_hp_ivs`` uses, so a paste that already states a deliberate spread is left alone.
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        stripped = strip_nickname(lines[0])
        if stripped != lines[0].rstrip():
            nicknames_stripped[0] += 1
            lines[0] = stripped
        moves = [l.strip()[2:].strip() for l in lines if l.strip().startswith("- ")]
        hp_move = next((m for m in moves if "hiddenpower" in m.lower().replace(" ", "")), None)
        iv_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("IVs:")), None)
        if hp_move is not None:
            existing = lines[iv_idx].strip() if iv_idx is not None else ""
            # "absent, or every stated IV is 31" — anything else is deliberate, leave it.
            stated = re.findall(r"(\d+)\s+\w+", existing)
            if not stated or all(int(v) == 31 for v in stated):
                hp_type = _hp_type(hp_move)
                if hp_type in GEN3_HP_IVS:
                    new_line = _iv_line(hp_type)
                    if iv_idx is not None:
                        if new_line:
                            lines[iv_idx] = new_line
                        else:
                            lines.pop(iv_idx)
                    elif new_line:
                        # After the EVs/Nature block, before the moves — Showdown accepts any
                        # order, but this is where a human paste puts it.
                        first_move = next(
                            (i for i, l in enumerate(lines) if l.strip().startswith("- ")),
                            len(lines))
                        lines.insert(first_move, new_line)
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory to write *.gen3ou_team files into")
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--limit", type=int, default=None,
                    help="export at most N teams (deterministic: the first N by sha1)")
    ap.add_argument("--manifest", default=None, help="write a JSON manifest here")
    args = ap.parse_args(argv)

    loader = TeamLoader()
    teams = loader.get_all_teams()
    n_sample, n_other = len(loader.get_sample_teams()), len(loader.get_other_teams())
    print(f"[export] pool: {len(teams)} teams ({n_sample} sample, {n_other} other)")

    from utils.bridge.team_validator import validate_teams_locally
    validations = validate_teams_locally(args.format, teams)
    valid = [t for t, v in zip(teams, validations) if v.get("valid")]
    print(f"[export] locally valid for {args.format}: {len(valid)} / {len(teams)}")

    parser = _Parser()
    rows = []
    for text in valid:
        sha = hashlib.sha1(text.strip().encode()).hexdigest()[:10]
        fixed = fix_team_text(text, parser)
        rows.append((sha, fixed, fixed != text.strip() + "\n"))
    rows.sort(key=lambda r: r[0])
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    n_fixed = 0
    for sha, fixed, changed in rows:
        n_fixed += bool(changed)
        with open(os.path.join(args.out, f"{sha}.{args.format}_team"), "w") as f:
            f.write(fixed)
    print(f"[export] wrote {len(rows)} files to {args.out} "
          f"({n_fixed} had their Hidden-Power IVs rewritten; "
          f"{nicknames_stripped[0]} nickname lines stripped)")

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({
                "format": args.format,
                "pool_total": len(teams),
                "pool_sample": n_sample,
                "pool_other": n_other,
                "locally_valid": len(valid),
                "exported": len(rows),
                "hp_iv_rewritten": n_fixed,
                "nickname_lines_stripped": nicknames_stripped[0],
                "team_sha1_10": [r[0] for r in rows],
            }, f, indent=1)
        print(f"[export] manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
