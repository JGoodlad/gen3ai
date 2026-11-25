"""X72's probe: the PLAYER of ONE cell, run BY PATH on the play checkout's code.

    PYTHONPATH=<play checkout>/src python -P <this file> --want want.json --out chunk.json --checkout <play checkout>

:mod:`main.x72.probe` exists at no commit X71 was read at, and the probe's three new cells must be the same game as
X71's twelve banked ones, so its driver runs THIS file by path (``-P``: the file's directory stays off ``sys.path``)
with the play checkout's ``src`` on ``PYTHONPATH``. It imports only the standard library and the play checkout's
``agents.training.untaught_meter``; nothing of ``main.x72``.

The game is X71's reader's (``read_x71.py`` ``play_chunk``, the calls restated and pinned by ``probe_test.py``): the
ref on the five pinned offense teams against the target, greedy against greedy (``stochastic=False``), the meter's
default CPU compute, one seed. Before any game it refuses (exit 3, nothing written) when

* the meter it imported is not the play checkout's (a missing ``PYTHONPATH`` would play another commit's game),
  or the play checkout is not at the planned playing commit;
* the play checkout's team files are not the ones X71's banked cells were played on (index, key and sha1 of each);
* the ref or the target is not the planned file (sha256).

The chunk is ``want`` plus ``counts`` (one cell a team), ``transport`` and ``game_rows``, written atomically: X71's
chunk with the team paths, the round and the playing commit added.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXIT_OK, EXIT_REFUSED = 0, 3


class PlayRefused(Exception):
    pass


def play(want: Dict[str, Any], *, checkout: Optional[str] = None, um: Any = None) -> Dict[str, Any]:
    """Play ``want``'s cell and return the chunk body. ``um`` replaces the meter in a test."""
    if um is None:
        from agents.training import untaught_meter as um
    if checkout is not None:
        src = os.path.realpath(os.path.join(checkout, "src")) + os.sep
        if not os.path.realpath(um.__file__).startswith(src):
            raise PlayRefused(f"the meter in use is {um.__file__}, not {src}...: set PYTHONPATH to the play "
                              f"checkout's src")
        head = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout.strip() != want["play_commit"]:
            raise PlayRefused(f"{checkout} is at {head.stdout.strip() or 'no commit'}, not the planned playing "
                              f"commit {want['play_commit']}")
    where = f"{want['target_tag']}/{want['cell']}"
    teams = um.team_slices(list(want["team_paths"]), prefix="X", source=f"{where}: the responders' recorded pin")
    got = [[x.index, x.key, x.sha1] for x in teams]
    if got != [list(t) for t in want["teams"]]:
        raise PlayRefused(f"{where}: the play checkout's team files are {got}, not the ones the banked cells were "
                          f"played on {want['teams']}")
    for key, sha in (("path", "sha256"), ("target", "target_sha256")):
        seen = um.file_sha256(want[key])
        if seen != want[sha]:
            raise PlayRefused(f"{where}: {want[key]} has sha256 {seen[:12]}, planned {str(want[sha])[:12]}")
    ref = um.resolve_ref(want["path"], label=f"{want['target_tag']}_{want['cell']}")
    target = um.resolve_ref(want["target"], label="target", role="opponent")
    log: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    cells = um.play_cells([ref], list(teams), target, games_per_team=int(want["games_per_team"]),
                          seed=int(want["seed"]), stochastic=False, game_log=log)
    rows = {team: games for (_ref, team), games in log.items()}
    return dict(want, counts={k: asdict(c) for k, c in cells[ref.label].items()},
                transport=um.cells_transport(cells), game_rows=rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="probe_play.py", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--want", required=True, help="the cell to play (main.x72.probe writes it)")
    ap.add_argument("--out", required=True, help="the chunk file; never overwritten")
    ap.add_argument("--checkout", required=True, help="the play checkout whose meter must be the one imported")
    a = ap.parse_args(argv)
    out = Path(a.out)
    try:
        if out.exists():
            raise PlayRefused(f"{out} exists: a banked cell is never replayed over")
        body = play(json.loads(Path(a.want).read_text()), checkout=a.checkout)
    except PlayRefused as exc:
        print(f"[x72 probe] REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, sort_keys=True) + "\n")
    os.replace(tmp, out)
    wins = sum(int(c["wins"]) for c in body["counts"].values())
    fin = sum(int(c["finished"]) for c in body["counts"].values())
    print(f"[x72 probe] banked {out.name}: {wins}/{fin} finished games won")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
