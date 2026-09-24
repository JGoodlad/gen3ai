"""Tracker-equivalence fuzz against the RUST CORE — the ``turn_delta_fold_equivalence_fuzz_test``
shape re-pointed at the core (``gen3_core_parity_trackers_v1``, the Rust Core Program's M3).

That fuzz proved the event-sourced ``TurnDelta`` fold value-identical to its snapshot-diff ancestor
on fresh real battles. Its successor question is whether the CORE's per-decision trackers — folded
on the version from one side's stream — equal what training builds: the real ``EpisodeTracker``
driven as ``Gen3Env.embed_battle`` drives it, the α/β label and the win-indicator reward (slice T,
:mod:`agents.battle.rust_core_parity_trackers`), on NEW battles every run: seeded-random players over
the rust bridge, teams from the pool, the fold fuzz's mechanic-dense ``MIXED_TEAM`` (Haze, Pain
Split, Explosion, Spikes/Roar, boosts, status) and the PROCEDURAL generator (Smogon-derived teams
outside the pool). Every battle also runs slices E and V on the same replay.

Coverage is read off the core's NATIVE record (``record::Window``): denials, Baton Pass entries,
drags, called moves, multi-faint windows, hazard KOs — so a run that never reached a corner says so.

Run directly (no server — the rust bridge):
    python src/agents/battle/rust_core_trackers_fuzz_test.py [--minutes 10] [--procedural 0.3]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
Exit 1 on any divergence.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
import time
from typing import Counter, Iterable, Optional, Tuple


def _coverage(res_trackers, cov: Counter) -> None:
    """Corner paths the native record shows, per viewer window."""
    for viewer in res_trackers or []:
        for cap in viewer:
            w = cap.get("window")
            if not w:
                continue
            faints = 0
            for a in w:
                k = a["kind"]
                cov[f"action:{k}"] += 1
                if k == "denied":
                    cov["denied"] += 1
                if k == "cant" and not a.get("then_moved"):
                    cov["refused"] += 1
                if k == "move" and a.get("called_by"):
                    cov["called_move"] += 1
                if k == "move" and a.get("pursuit_on_switch"):
                    cov["pursuit_on_switch"] += 1
                if k == "switch" and isinstance(a.get("entry"), dict):
                    cov["entry:" + next(iter(a["entry"]))] += 1
                for e in a["effects"]:
                    if e[1] == "faint":
                        faints += 1
                        cov[f"faint_cause:{e[3][0]}"] += 1
            if faints >= 2:
                cov["multi_faint_window"] += 1


def _battles(rng: random.Random, n: int, procedural: float) -> Iterable[Tuple[int, Optional[Tuple[str, str]]]]:
    from agents.battle import rust_core_parity as P
    from agents.training.turn_delta_fold_equivalence_fuzz_test import MIXED_TEAM
    from utils.team_loader import TeamLoader

    pool = TeamLoader().get_all_teams()
    proc = P.procedural_teams(max(2, int(2 * n * procedural) + 2), rng.randrange(1 << 30)) if procedural > 0 else []
    for i in range(n):
        # the key recipe's sim seed is [11+key, 22+key, 33+key, 44+key], each a u16
        key = rng.randrange(10_000, 65_000)
        u = rng.random()
        if proc and u < procedural:
            yield key, (proc.pop(), proc.pop()) if len(proc) >= 2 else None
        elif u < procedural + 0.25:
            yield key, (MIXED_TEAM, pool[key % len(pool)])
        else:
            yield key, None


def main(argv: Optional[list] = None) -> int:
    from agents.battle import rust_core_parity as P
    from agents.battle import rust_core_parity_trackers as T
    from agents.battle import rust_core_parity_views as V

    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--procedural", type=float, default=0.3, help="share of battles on procedural teams")
    ap.add_argument("--batch", type=int, default=8, help="battles per core_events process")
    ap.add_argument("--seed", type=int, default=None, help="default: a fresh seed every run")
    args = ap.parse_args(argv)
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (1 << 31)
    rng = random.Random(seed)
    print(f"Tracker-equivalence fuzz (core vs EpisodeTracker) — seed {seed} — budget {args.minutes:.1f} min", flush=True)
    events, views, trackers = P.Census(), V.ViewCensus(), T.TrackerCensus()
    cov: Counter = collections.Counter()
    deadline = time.time() + args.minutes * 60
    played = skips = 0
    while time.time() < deadline:
        batch = []
        for key, teams in _battles(rng, args.batch, args.procedural):
            try:
                lv = P.play(key, tag="Tf", teams=teams)
            except Exception as e:                                      # noqa: BLE001 - infra, counted
                skips += 1
                if skips <= 5:
                    print(f"  [infra-skip #{skips}] {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            P.compare_live(lv, events)
            batch.append(lv.recorded)
        played += len(batch)
        P.check_battles(batch, events, views=views, trackers=trackers,
                        on_result=lambda _b, res: _coverage(res.get("trackers"), cov))
        bad = sum(events.divergences.values()) + sum(views.divergences.values()) + sum(trackers.divergences.values())
        print(f"  battles={played} decisions={trackers.decisions} rows={trackers.window_rows} "
              f"divergences={bad} refused={len(trackers.refused) + len(events.refused)} | "
              f"denied={cov['denied']} refused-actions={cov['refused']} bp={cov['entry:batonpass']} "
              f"drag={cov['action:drag']} called={cov['called_move']} multi-faint={cov['multi_faint_window']}",
              flush=True)
    print(events.render())
    print(views.render())
    print(trackers.render())
    print("coverage:", json.dumps(dict(sorted(cov.items()))))
    clean = not (events.divergences or views.divergences or trackers.divergences or events.refused or trackers.refused)
    if skips:
        print(f"infra skips: {skips}")
    return 0 if clean and trackers.decisions > 0 and skips <= max(1, played // 4) else 1


if __name__ == "__main__":
    sys.exit(main())
