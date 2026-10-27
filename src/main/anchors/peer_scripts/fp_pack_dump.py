"""Foul Play's OWN pack of each team file, for ``main.anchors.frontend_pack``. Runs under FOUL PLAY's interpreter.

stdin: ``{"checkout": "<foul-play checkout>", "files": ["<abs path>", ...]}``; stdout: ``{path: packed | null}`` where
``null`` is a file Foul Play's ``export_to_packed`` raises on (it would crash at that draw). STDLIB + Foul Play only:
the script imports ``fp.teams.team_converter`` from the checkout, so what it prints is the string Foul Play sends in
``/utm`` for that file, never a re-implementation of it."""
import json
import os
import sys


def main() -> int:
    req = json.loads(sys.stdin.read())
    sys.path.insert(0, req["checkout"])
    os.chdir(req["checkout"])
    from fp.teams.team_converter import export_to_packed

    out = {}
    for path in req["files"]:
        try:
            with open(path, "r") as fh:
                out[path] = export_to_packed(fh.read())
        except Exception:                               # noqa: BLE001 - the peer would crash on this file
            out[path] = None
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
