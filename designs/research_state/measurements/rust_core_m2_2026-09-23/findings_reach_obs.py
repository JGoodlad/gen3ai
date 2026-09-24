"""Does each poke-env READING finding reach the TRAINING INPUT (the 2501-dim obs)?

For each finding, one minimal protocol excerpt is fed to a `Gen3Battle` (the live player's dispatch,
`offline_feed`), and the per-block ENCODERS the training obs is assembled from are run on the
`LiveView` poke-env builds and on the same `LiveView` with the finding's field set to the TRUTH. A
byte difference means the finding reaches the obs. Printed per finding; nothing is written.

    PYTHONPATH=src python findings_reach_obs.py
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agents.battle.offline_feed import feed_line, new_battle
from agents.observation.active_context import ActiveContextEncoder
from agents.observation.state_encoder import load_mappings

TEAM = ("Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]"
        "Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||")
BASE = ["|player|p1|me||", "|player|p2|foe||", "|teamsize|p1|2", "|teamsize|p2|3", "|gen|3",
        "|tier|[Gen 3] OU", "|start", "|switch|p1a: Metagross|Metagross|301/301",
        "|switch|p2a: Zapdos|Zapdos|100/100", "|turn|1"]


def battle(lines):
    b = new_battle("p1", {"p1": "me", "p2": "foe"}, packed_team=TEAM)
    for ln in BASE + lines:
        feed_line(b, ln)
    return b


def opp_mon(live, sp):
    return next(m for m in live.opp.mons if m.species == sp)


def main() -> None:
    maps = load_mappings()
    act = ActiveContextEncoder(maps.get("moves"))

    # PE-V10 — a fainted active mon keeps its stages (poke-env) vs cleared (the sim).
    b = battle(["|-boost|p2a: Zapdos|spa|1", "|faint|p2a: Zapdos"])
    live = b.live_view()
    active = live.opp.active
    print("PE-V10: LiveView opp.active =", getattr(active, "species", None),
          "boosts", getattr(active, "boosts", None))
    if active is not None:
        truth = dataclasses.replace(active, boosts={})
        d = np.flatnonzero(act.encode(active) != act.encode(truth))
        print("  active_context bytes differing:", d.tolist())

    # PE-V16 — Flash Fire ended by the holder's own Fire move (poke-env) vs kept (the sim).
    b = battle(["|switch|p2a: Houndoom|Houndoom, M|100/100", "|-start|p2a: Houndoom|ability: Flash Fire",
                "|move|p2a: Houndoom|Flamethrower|p1a: Metagross"])
    live = b.live_view()
    active = live.opp.active
    print("PE-V16: LiveView opp.active =", active.species, "volatiles", dict(active.volatiles))
    truth = dataclasses.replace(active, volatiles={**dict(active.volatiles), "flashfire": 0})
    d = np.flatnonzero(act.encode(active) != act.encode(truth))
    print("  active_context bytes differing:", d.tolist())

    # PE-R1b — the toxic counter of a mon that entered after the residual, one ahead of the sim.
    b = battle(["|-status|p2a: Zapdos|tox", "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
                "|switch|p2a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Zapdos|Zapdos|94/100 tox", "|turn|3"])
    live = b.live_view()
    z = opp_mon(live, "zapdos")
    print("PE-R1b: poke-env status_counter", z.status_counter, "(the sim's stage: 0 — no residual since re-entry)")
    print("  the pokemon block's toxic slot = min(ctr, 8) / 8 → poke-env", min(z.status_counter, 8) / 8.0,
          "truth", 0.0)


if __name__ == "__main__":
    main()
