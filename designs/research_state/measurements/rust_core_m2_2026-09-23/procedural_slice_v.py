"""Slice V (with the core's present() column and board audit) on PROCEDURAL gen3ou teams.

The pool reaches some reading rules rarely or never (R2 / R4 / the locked request appeared only on
`ou_random_teams.js`'s teams, `one_sided_view.md` §4b), and Flash Fire holders are uncommon in the
719-team pool. This plays N battles between procedural teams (Smogon-derived, validated by
Showdown's own TeamValidator, reproducible from `--seed`) over the production rust bridge, runs
slices E + V on them, and prints both censuses — including `reading-vs-truth rules fired`, the
count of facts where the READING differs from the engine under a named rule (V10 / R1b / V16).

    PYTHONPATH=src python procedural_slice_v.py --battles 200 --seed 23
"""

from __future__ import annotations

import argparse
import logging

from agents.battle import rust_core_parity as P
from agents.battle import rust_core_parity_views as V


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--key0", type=int, default=9000)
    a = ap.parse_args()
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    teams = P.procedural_teams(2 * a.battles, a.seed)
    census, views = P.Census(), V.ViewCensus()
    lives = []
    for i in range(a.battles):
        lives.append(P.play(a.key0 + i, tag="Pv", teams=(teams[2 * i], teams[2 * i + 1])))
    for lv in lives:
        P.compare_live(lv, census)
    P.check_battles([lv.recorded for lv in lives], census, views=views)
    print(census.render())
    print(views.render())
    return 0 if not views.divergences and not views.refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
