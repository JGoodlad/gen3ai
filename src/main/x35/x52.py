"""X52, THE PRACTICE CEILING (``design_beat_metamon_path.md`` §6.2; registered in the X35 launch sheet's §0, AM-X52,
2026-10-22, before this module existed). It BUILDS one lever and CHECKS its team files; it never launches or plays.

THE ARM ``x52`` (production side, side-pinned at X35's production pin, seeds 3501-3502): C0's argv token for token
(``main.x35.kit.build_argv``) at +5M instead of +20M, plus the ONE lever ``--trainee-teams <the 36 practised
files>``. Our side pilots ONLY the practised half; the opponent side keeps the pool (facing is not the deficit).

THE FILES are the registered halves in ``designs/research_state/measurements/x52_practice_ceiling_2026-10-22/``
(``draw_x52_teams.py --check`` reproduces them). The trainer opens ``--trainee-teams`` paths against the launch CWD
(main's checkout, ``launch_x35.sh``; a pinned child keeps it), so the lever names them REPO-RELATIVE and every build
first verifies the files against the manifest's sha256 (F-X52-4): a moved, edited or missing file is a refusal, never
a different practice set.

    python -m main.x35.x52 verify      # exit 1 unless the 36 practised files match the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils.paths import repo_root

#: The X52 arm and its side (``main.x35.kit.ARMS`` holds the Arm; ``main.x35.batch.X52_ARMS`` side-pins it).
X52_ARMS: Dict[str, str] = {"x52": "production"}
#: AM-X52-c: P* + 5M, seeds 3501 and 3502 (C0's first two, at C0's pin).
L_X52 = 5_000_000
X52_SEEDS = (3501, 3502)
#: Repo-relative: the registered halves and their manifest (AM-X52-a).
HALVES_DIR = "designs/research_state/measurements/x52_practice_ceiling_2026-10-22"
MANIFEST = f"{HALVES_DIR}/x52_teams_manifest.json"
PRACTISED_DIR = f"{HALVES_DIR}/teams_practised"
MANIFEST_SCHEMA = "x52_practice_halves_v1"
N_PRACTISED = 36


class X52Error(ValueError):
    """X52's practised files do not match their registration (named; the launch HOLDs on it)."""


def practised_files(repo: Optional[Path] = None) -> List[str]:
    """The 36 practised files, REPO-RELATIVE and sorted, after checking them against the manifest: the set of files
    on disk is exactly the manifest's practised ids, and each file's sha256 is the manifest's."""
    root = Path(repo) if repo is not None else repo_root()
    mpath = root / MANIFEST
    try:
        man = json.loads(mpath.read_text())
    except (OSError, ValueError) as exc:
        raise X52Error(f"{MANIFEST}: unreadable ({exc})") from exc
    if man.get("schema") != MANIFEST_SCHEMA:
        raise X52Error(f"{MANIFEST}: schema {man.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    want = {f"{i}.txt": t["sha256"] for i, t in (man.get("teams") or {}).items() if t.get("half") == "practised"}
    if len(want) != N_PRACTISED:
        raise X52Error(f"{MANIFEST} names {len(want)} practised teams, not {N_PRACTISED}")
    d = root / PRACTISED_DIR
    have = {p.name for p in d.glob("*.txt")} if d.is_dir() else set()
    if have != set(want):
        raise X52Error(f"{PRACTISED_DIR}: files {sorted(have ^ set(want))} differ from the manifest's practised set")
    bad = [n for n in sorted(want) if hashlib.sha256((d / n).read_bytes()).hexdigest() != want[n]]
    if bad:
        raise X52Error(f"{PRACTISED_DIR}: sha256 differs from the manifest for {bad}")
    return [f"{PRACTISED_DIR}/{n}" for n in sorted(want)]


def lever(repo: Optional[Path] = None) -> List[str]:
    """X52's ONE lever (AM-X52-b), verified."""
    return ["--trainee-teams", ",".join(practised_files(repo))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="exit 1 unless the practised files match the manifest")
    ap.parse_args(argv)
    try:
        files = practised_files()
    except X52Error as exc:
        print(f"X52 FAIL: {exc}")
        return 1
    print(f"X52 OK: {len(files)} practised files match {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
