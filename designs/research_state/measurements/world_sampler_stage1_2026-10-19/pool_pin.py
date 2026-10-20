"""C_pool's POOL exactly as X26's PIN packed it (PLAY side, ``run_s1.sh pin``): the trainer's own path
(``Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams``, Tier 1's ``pool_teams.load_pool``) run at the pin.

Why not the builder's own: the Hidden Power refactor of 2026-10-19 (``c9493053b``, and ``f20526571``'s Ghost fill)
changed the IVs the packing path FILLS for a typed Hidden Power, so the builder commit packs some pool sets with other
30 / 31 IV patterns than the teams X26 actually met (the bank's true teams, packed at the pin). POOL gates a candidate
WHOLE (its seen mons' IVs included), so a re-packed true team can fail the gate by a stat point and the ceiling would
lose the truth. The pin's packing is the population the bank's opponents were drawn from.

    run_s1.sh pin 1 4 30m pool_pin.py        -> <archive>/pool/pool_pin.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_common as E  # noqa: E402


def main():
    if os.environ.get("WS1_KIND") != "pin":
        raise SystemExit("run at the PIN checkout (run_s1.sh pin)")
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(os.environ["WS1_CHECKOUT_DIR"]):
        raise SystemExit(f"main imported from {_m.__file__}, not the pin checkout")
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    raw = TeamLoader().get_all_teams()
    teams = list(Gen3Teambuilder(raw).packed_teams)
    if len(teams) != len(raw):
        raise SystemExit(f"the validator dropped {len(raw) - len(teams)} of {len(raw)} pool teams")
    out = {"pin_head": os.environ["WS1_CHECKOUT_HEAD"], "n": len(teams), "teams": teams,
           "sha256": hashlib.sha256("\n".join(teams).encode()).hexdigest()}
    p = E.d("pool") / "pool_pin.json"
    p.write_text(json.dumps(out) + "\n")
    print(f"[s1 pool] {len(teams)} teams ({len(set(teams))} distinct), sha256 {out['sha256'][:12]} -> {p}")


if __name__ == "__main__":
    main()
