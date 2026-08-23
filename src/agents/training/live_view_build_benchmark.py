"""ORDER-ALTERNATED, SAME-PROCESS A/B of `LiveView.from_battle` on a REAL, FROZEN board.

This is a **benchmark, not a pass/fail test** (no ``test_*`` funcs, so ``pytest`` imports it and
collects nothing). Run it directly.

`LiveView.from_battle` is the largest single item in per-decision worker CPU — **17%**, once
`gen3_live_view_memo_v1` collapsed the five per-decision rebuilds into one. Optimising it needs
an instrument the trainer-turn benchmark cannot be: that one walks a *fresh random battle* per
invocation, so two consecutive runs profile different boards and a 12-mon turn-40 board reads
faster than a 12-mon turn-65 board on a strictly faster tree. **That mistake was made during
`gen3_live_view_build_micros_v1` and read as a regression**, which is why this file exists.

So: capture ONE real mid-game board, freeze it, and rebuild the view N times under BOTH
implementations — the live one and a verbatim copy of the pre-optimization code (`_ref_*` below)
— alternating which arm goes first each round. Neither the board, the machine load, nor cache
warmth can then favour an arm. It reports the wall-clock ratio AND a load-free
`sys.setprofile` call count per build, which is the primary on a busy box.

⚠️ It first CHECKS that the two arms build field-identical mons, so a "speedup" that changed an
answer refuses to be reported as a speedup. The `_ref_*` copy is deliberately a DUPLICATE and is
expected to drift out of date: when it does, the agreement check fails loudly and you either
update the reference or delete the arm — it will not silently measure two copies of the same
code.

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/live_view_build_benchmark.py [--reps 2500] [--rounds 6]
                                                            [--turn 12] [--profile]
"""
from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import pstats
import random
import statistics
import sys
import time
from typing import Optional

from poke_env import AccountConfiguration
from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

import agents.training.trainer_turn_benchmark as ttb
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import (LiveMove, LivePokemon, LiveSide, LiveView, _UNKNOWN_ITEMS,
                                     _fold_weather)
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

CAPTURED = []


# --------------------------------------------------------------------------- #
# REFERENCE arm — verbatim pre-optimization code (git `fd69af7`).              #
# --------------------------------------------------------------------------- #
def _ref_enum_name(value) -> Optional[str]:
    return value.name.lower() if value is not None else None


def _ref_id(value) -> Optional[str]:
    if value is None:
        return None
    return value.name.lower().replace("_", "")


def _ref_entry(mv):
    if mv._id in GenData.from_gen(mv.gen).moves:
        return GenData.from_gen(mv.gen).moves[mv._id]
    elif mv._id.startswith("z") and mv._id[1:] in GenData.from_gen(mv.gen).moves:
        return GenData.from_gen(mv.gen).moves[mv._id[1:]]
    elif mv._id in {"recharge", "fight"}:
        return {"pp": 1, "type": "normal", "category": "Special", "accuracy": 1}
    raise ValueError("Unknown move: %s" % mv._id)


def _ref_max_pp(mv) -> int:
    max_pp = _ref_entry(mv)["pp"] * 8 // 5
    if mv.gen >= 5 and mv._from_transform:
        return min(5, max_pp)
    elif mv.gen < 3:
        max_pp = min(max_pp, 61)
    return max_pp


def _ref_from_pokemon(mon, active: bool, is_own: bool = False) -> LivePokemon:
    item = mon.item if mon.item not in _UNKNOWN_ITEMS else None
    moves = tuple(
        LiveMove(id=mid, current_pp=int(mv.current_pp), max_pp=int(_ref_max_pp(mv)))
        for mid, mv in sorted(mon.moves.items())
    )
    ivs = tuple(mon.ivs) if (is_own and mon.ivs is not None) else None
    evs = tuple(mon.evs) if (is_own and mon.evs is not None) else None
    nature = mon.nature if is_own else None
    consumed = mon.consumed_item
    consumed_item = to_id_str(consumed) if consumed else None
    return LivePokemon(
        species=mon.species,
        active=active,
        fainted=bool(mon.fainted),
        revealed=bool(mon.revealed),
        hp_fraction=float(mon.current_hp_fraction),
        status=_ref_enum_name(mon.status),
        types=tuple(_ref_enum_name(t) for t in mon.types if t is not None),
        moves=moves,
        item=item,
        ability=mon.ability,
        boosts={k: v for k, v in mon.boosts.items() if v},
        volatiles={_ref_id(e): int(cnt) for e, cnt in mon.effects.items()},
        base_stats=dict(mon.base_stats),
        ivs=ivs,
        evs=evs,
        nature=nature,
        spread_known=bool(is_own),
        consumed_item=consumed_item,
        status_counter=int(getattr(mon, "status_counter", 0) or 0),
        protect_counter=int(getattr(mon, "protect_counter", 0) or 0),
        stats=dict(mon.stats) if getattr(mon, "stats", None) else {},
        current_hp=(int(mon.current_hp) if getattr(mon, "current_hp", None) is not None else None),
        max_hp=(int(mon.max_hp) if getattr(mon, "max_hp", None) is not None else None),
    )


def _ref_from_battle(battle) -> LiveView:
    """Verbatim `LiveView.from_battle`, with `_ref_from_pokemon` / `_ref_enum_name` swapped in."""
    role = battle._player_role
    opp_role = "p2" if role == "p1" else "p1"
    sizes = getattr(battle, "_team_size", {}) or {}

    def side(team, conditions, declared_role, active_mon, is_own) -> LiveSide:
        built = {}
        active = None
        for raw in team.values():
            is_active = raw is active_mon
            lm = _ref_from_pokemon(raw, active=is_active, is_own=is_own)
            built[id(raw)] = lm
            if is_active:
                active = lm
        return LiveSide(
            team_size=int(sizes.get(declared_role, len(built))),
            active=active,
            mons=tuple(built.values()),
            side_conditions={_ref_enum_name(k): v for k, v in (conditions or {}).items()},
        )

    if hasattr(battle, "live_weather"):
        weather = battle.live_weather()
    else:
        weather = _fold_weather(getattr(battle, "events", ()), battle.turn)
    return LiveView(
        turn=battle.turn, weather=weather,
        ours=side(battle.team, battle.side_conditions, role, battle.active_pokemon, True),
        opp=side(battle.opponent_team, battle.opponent_side_conditions, opp_role,
                 battle.opponent_active_pokemon, False),
        battle_tag=battle.battle_tag, finished=bool(battle.finished),
        won=battle.won, lost=battle.lost)


class _Capture(Player):
    def __init__(self, *args, turn: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = turn

    def choose_move(self, battle):
        if battle.turn >= self._target and not CAPTURED:
            CAPTURED.append(battle)
        return self.choose_random_move(battle)


async def _capture(turn: int, seed: int) -> None:
    random.seed(seed)
    pool = ttb._team_pool()
    ts = int(time.time()) % 100000
    p1 = _Capture(turn=turn, battle_format=ttb.BATTLE_FORMAT, team=Gen3Teambuilder(pool),
                  account_configuration=AccountConfiguration(f"LVz{ts}", "pw"),
                  server_configuration=LocalhostServerConfiguration, start_listening=False,
                  battle_class=Gen3Battle)
    p2 = RandomPlayer(battle_format=ttb.BATTLE_FORMAT, team=Gen3Teambuilder(pool),
                      account_configuration=AccountConfiguration(f"LVo{ts}", "pw"),
                      server_configuration=LocalhostServerConfiguration, start_listening=False)
    tries = 0
    while not CAPTURED and tries < 12:
        await run_local_battles(p1, p2, 1, seed=[7, 11, 13, 17])
        tries += 1


def _mons(v: LiveView):
    return list(v.ours.mons) + list(v.opp.mons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--turn", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    asyncio.run(_capture(a.turn, a.seed))
    if not CAPTURED:
        print("no battle captured", file=sys.stderr)
        return 1
    battle = CAPTURED[0]
    n_mons = len(battle.team) + len(battle.opponent_team)
    n_moves = sum(len(m.moves) for m in list(battle.team.values())
                  + list(battle.opponent_team.values()))
    print(f"board: turn {battle.turn}, {n_mons} mons, {n_moves} revealed moves "
          f"({len(battle.team)} ours / {len(battle.opponent_team)} opp)")

    # --- agreement check: the two arms must build the SAME per-mon view --------
    new_v, ref_v = LiveView.from_battle(battle), _ref_from_battle(battle)
    bad = [(x, y) for x, y in zip(_mons(new_v), _mons(ref_v)) if x != y]
    if bad or len(_mons(new_v)) != len(_mons(ref_v)):
        print(f"ARMS DISAGREE on {len(bad)} mons — NOT a valid measurement", file=sys.stderr)
        for x, y in bad[:2]:
            print(f"  new={x}\n  ref={y}", file=sys.stderr)
        return 2
    print(f"arms agree on all {len(_mons(new_v))} mons ✓")

    for _ in range(300):                       # warm both arms
        LiveView.from_battle(battle)
        _ref_from_battle(battle)

    def time_arm(fn) -> float:
        t0 = time.perf_counter()
        for _ in range(a.reps):
            fn(battle)
        return (time.perf_counter() - t0) / a.reps * 1e3

    ratios, news, refs = [], [], []
    for r in range(a.rounds):
        if r % 2 == 0:                          # ORDER-ALTERNATED
            new = time_arm(LiveView.from_battle); ref = time_arm(_ref_from_battle)
        else:
            ref = time_arm(_ref_from_battle); new = time_arm(LiveView.from_battle)
        news.append(new); refs.append(ref); ratios.append(ref / new if new else 0.0)
        print(f"  round {r} ({'new,ref' if r % 2 == 0 else 'ref,new'}): "
              f"new {new:.4f} ms   ref {ref:.4f} ms   ratio {ratios[-1]:.3f}x")
    print(f"\n  MEDIAN: new {statistics.median(news):.4f} ms   "
          f"ref {statistics.median(refs):.4f} ms   RATIO {statistics.median(ratios):.3f}x")

    # --- LOAD-FREE primary: Python calls per build, both arms, same board -----
    counter = [0]

    def _count(frame, event, arg):
        if event == "call":
            counter[0] += 1

    def calls_for(fn) -> float:
        counter[0] = 0
        sys.setprofile(_count)
        for _ in range(200):
            fn(battle)
        sys.setprofile(None)
        return counter[0] / 200.0

    c_new, c_ref = calls_for(LiveView.from_battle), calls_for(_ref_from_battle)
    print(f"  PYTHON CALLS / build (sys.setprofile, load-free): new {c_new:.1f}  ref {c_ref:.1f}"
          f"   ({(c_new - c_ref) / c_ref * 100:+.1f}%)")

    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(a.reps):
            LiveView.from_battle(battle)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(a.top)
        print(s.getvalue())
    return 0


if __name__ == "__main__":
    # A benchmark on a busy box reports a confidently wrong number — say so up front. (The RATIO
    # here is far more load-robust than the absolutes, because both arms run interleaved on the
    # same board; the call-count line is load-free outright.)
    from utils.contention import warn_if_contended
    warn_if_contended("live-view build benchmark")
    sys.exit(main())
