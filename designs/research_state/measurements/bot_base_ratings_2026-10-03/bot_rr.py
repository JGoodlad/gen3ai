"""Rustboro-era BOT BASE RATINGS — the play half: a MIRRORED, PER-GAME-SEEDED bot-vs-bot round robin.

    export PYTHONPATH=<checkout>/src
    python bot_rr.py play --pairs 2000 --batch-pairs 100 --workers 12 --out <dir>/ledger
    python bot_rr.py plan --pairs 2000 --batch-pairs 100 --out <dir>/ledger     # what is left to play

WHAT IS PLAYED. Every unordered pair of the nine eval-roster bots (``eval_opponent_names()``, the bots the
ladder pins), ``--pairs`` mirrored PAIRS per edge (2 games each), in batches of ``--batch-pairs``.

* TEAMS — the trainee's eval team distribution (``rust_eval.build.eval_builders``' trainee builder: the
  default pool, 10 % sample-team bias), drawn by the Rust eval core's per-game seed rule
  (``rust_eval.seeds.draw_team`` at the game's key; seat A draws with side ``trainee``, seat B with
  ``opponent``, two independent builder instances).
* MIRROR — ``gen3_mirrored_pairs_v1`` (``seeds.pair_game``): games ``2k`` / ``2k+1`` share ONE key (the same
  two teams, battle seed and bot streams) and the second game hands the teams over.
* SEAT — the mirror cancels team luck, not the seat. So the SEAT ALTERNATES BY PAIR: in pair ``k`` bot A
  sits p1 when ``k`` is even, bot B when ``k`` is odd (both games of a pair keep one seating). Over an even
  number of pairs each bot sits p1 in exactly half the games of the edge.
* BOT STREAMS — the p2 bot's choice / protect streams are re-seeded per game EXACTLY as the Rust eval core
  re-seeds a p2 bot route (``seeds.bot_stream_seeds`` = ``episode_bot_stream_seed(bot_route_seed(name),
  battle seed, k)``); the p1 bot (a seat the core has no bot route for) gets the same rule on route seed
  ``bot_route_seed(name) + P1_ROUTE_OFFSET``.
* TURN LIMIT — the core's stall rule: p1 FORFEITS at its first decision with ``turn >= stall_threshold()``,
  and ``trace_result.classify_result`` scores a game at or past the cap as a DRAW (timeout), as training does.
* SIM — the Rust Showdown port over the in-process bridge (``run_local_battles(impl="rust")``), one game per
  call at concurrency 1 with the game's own battle seed — the Lane H gate's Python path (``eval_worker``
  ``seed_rule = "per_game"``).

WHY NOT THE RUST ENV CORE ITSELF. The core's route table seats a scripted bot only at p2 (``opponents.rs``:
p1 is always answered by the caller); there is no p1 bot route. The bots here are the PYTHON bots
(``agents.opponents`` / ``poke_env.player.baselines``) — the reference implementation the Rust port is gated
against at 0 mismatches (``bots_gate_test``), driven by the Rust eval core's own seed rule.

ROWS. One §0b COUNT row (``agents.training.eval_ledger``, ``gen3_eval_count_row_v1``) per (edge x batch),
purpose ``anchor``; PLAYER = the edge's first bot in roster order (bot A), OPPONENT = bot B. A bot's ``sha256``
is the digest of its name + the sources of the bot modules at this commit (a bot is code, not a checkpoint).
Resumable: a (edge, batch) already on disk is skipped. Reproducible: every game is a pure function of
``(schedule seed, edge, batch, game index)`` and the code.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import os
import random
import sys
import time
from multiprocessing import get_context
from typing import Any, Dict, List, Tuple

ITEM_PREFIX = "botrr"
CYCLE_SEED_TAG = "gen3_botrr_cycle_seed_v1"
P1_ROUTE_OFFSET = 0x100
RUN_LABEL = "bot_base_ratings_2026-10-03"
MIRROR_RULE = "gen3_mirrored_pairs_v1+seat_alternating_by_pair"
EVAL_CORE = "rust_sim_bridge+python_bots(per_game_seed)"
TEAM_SOURCE = ("rust_eval.build.eval_builders trainee builder default_biased(bias_prob=0.1) for BOTH seats; "
               "seat A draws side 'trainee', seat B side 'opponent'")
BOT_SOURCES = ("agents/opponents.py", "poke_env/player/baselines.py", "poke_env/player/random_player.py",
               "agents/training/eval_roster.py")


def roster() -> List[str]:
    from agents.training.eval_callback import eval_opponent_names

    return list(eval_opponent_names())


def edges() -> List[Tuple[str, str]]:
    return list(itertools.combinations(roster(), 2))


def item_key(a: str, b: str) -> str:
    return f"{ITEM_PREFIX}:{a}:{b}"


def cycle_seed(schedule_seed: int, key: str, batch: int) -> int:
    d = hashlib.blake2b(f"{CYCLE_SEED_TAG}:{int(schedule_seed)}:{key}:{int(batch)}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") & ((1 << 62) - 1)


def bot_block(name: str) -> Dict[str, Any]:
    from utils.paths import src_path

    h = hashlib.sha256(f"bot:{name}".encode())
    for rel in BOT_SOURCES:
        p = src_path(*rel.split("/"))
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(rel.encode())
                h.update(f.read())
    return {"id": f"bot:{name}", "sha256": h.hexdigest(), "path": "src/" + BOT_SOURCES[0], "run": "scripted_bots",
            "step": None, "rung": "scripted"}


def regime() -> Dict[str, Any]:
    from agents.training import eval_ledger as L
    from agents.training.rust_eval import seeds as SD
    from utils.rust_env import episode as EP

    # §0b's PLAYS vocabulary is greedy|sampled; a scripted bot draws from its own seeded streams -> "sampled".
    return L.with_regime_id({"play": "sampled", "opponent_play": "sampled", "mirrored": True,
                             "mirror_rule": MIRROR_RULE, "eval_core": EVAL_CORE,
                             "turn_limit": int(EP.stall_threshold()), "seed_rule": SD.SCHEMA,
                             "team_source": TEAM_SOURCE})


def batch_plan(pairs: int, batch_pairs: int) -> List[int]:
    if batch_pairs % 2:
        raise SystemExit("--batch-pairs must be EVEN (the seat alternates by pair inside a batch)")
    full, rem = divmod(pairs, batch_pairs)
    if rem:
        raise SystemExit("--pairs must be a multiple of --batch-pairs (a ragged batch would change a batch's length)")
    return [batch_pairs] * full


def done_units(out_dir: str, reg_id: str, schedule_seed: int) -> Dict[Tuple[str, int], int]:
    from agents.training import eval_ledger as L

    if not os.path.isdir(out_dir):
        return {}
    out = {}
    for r in L.read_rows(out_dir):
        if r["regime"]["regime_id"] == reg_id and r["seed"]["schedule_seed"] == schedule_seed:
            out[(r["seed"]["item_key"], int(r["seed"]["batch"]))] = int(r["pairs"]["n_pairs"])
    return out


# ------------------------------------------------------------------------------------------------ worker
_W: Dict[str, Any] = {}


class _Staged:
    """A teambuilder that yields whatever team was staged for the next game."""

    def __init__(self) -> None:
        self.team = ""

    def yield_team(self) -> str:
        return self.team


def _init_worker(out_dir: str) -> None:
    import torch

    torch.set_num_threads(1)
    from agents.training import eval_ledger as L
    from agents.training.rust_eval.build import eval_builders

    tb_a, _flat, _ = eval_builders(None, [])
    tb_b, _flat2, _ = eval_builders(None, [])
    _W["tb"] = (tb_a, tb_b)
    _W["writer"] = L.LedgerWriter(out_dir, writer_id=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}")
    _W["bots"] = {}
    _W["regime"] = regime()
    _W["blocks"] = {}
    try:
        from utils.git import get_git_hash

        _W["commit"] = get_git_hash()
    except Exception:                                                     # noqa: BLE001
        _W["commit"] = "unknown"


def _bot(name: str, seat: str):
    """One Python bot per (name, seat letter) per worker, built once (bridge play: no websocket)."""
    k = (name, seat)
    if k not in _W["bots"]:
        from poke_env.player.battle_order import ForfeitBattleOrder
        from poke_env.ps_client import LocalhostServerConfiguration
        from agents.training.eval_roster import build_eval_opponents
        from utils.rust_env import episode as EP

        st = _Staged()
        (_n, p), = build_eval_opponents(LocalhostServerConfiguration, st, [name], f"R{seat}{os.getpid() % 100000}",
                                        start_listening=False)
        p._team = st
        limit = int(EP.stall_threshold())
        orig = p.choose_move

        def choose_move(battle, _orig=orig, _p=p):
            # the core's stall rule: p1 forfeits at its first decision with turn >= the limit
            if getattr(_p, "_botrr_is_p1", False) and int(battle.turn or 0) >= limit:
                return ForfeitBattleOrder()
            return _orig(battle)

        p.choose_move = choose_move
        _W["bots"][k] = (p, st)
    return _W["bots"][k]


def _set_streams(p, seeds: Dict[str, int]) -> None:
    for stream, s in seeds.items():
        attr = {"choice": "_choice_rng", "protect": "_protect_rng"}[stream]
        if stream == "choice" or hasattr(p, attr):
            setattr(p, attr, random.Random(s))


def play_unit(unit: Tuple[str, str, int, int, int]) -> Dict[str, Any]:
    from agents.training import eval_ledger as L
    from agents.training import mirrored_pairs as MP
    from agents.training.rust_env_opponents import episode_bot_stream_seed
    from agents.training.rust_eval import seeds as SD
    from agents.training.trace_result import DRAW, LOSS, WIN, classify_result
    from utils.bridge.local_battle_runner import run_local_battles

    a, b, batch, n_pairs, schedule_seed = unit
    key = item_key(a, b)
    cseed = cycle_seed(schedule_seed, key, batch)
    pa, sta = _bot(a, "a")
    pb, stb = _bot(b, "b")
    tb_a, tb_b = _W["tb"]
    limit = int(_W["regime"]["turn_limit"])
    pa.reset_battles()
    pb.reset_battles()
    w = l = d = timeouts = unfinished = 0
    a_p1_games = 0
    pts: List[Any] = []
    teams: Dict[str, Dict[str, List[int]]] = {}

    def bump(tid: str, side: str, won: bool) -> None:
        t = teams.setdefault(tid, {"p": [0, 0], "o": [0, 0]})
        t[side][0] += 1
        t[side][1] += int(won)

    t_start, t0 = L.utc_now(), time.perf_counter()
    for g in range(2 * n_pairs):
        gk, swapped = SD.pair_game(cseed, key, g, True)
        words = SD.battle_seed(gk)
        t1, t2 = SD.draw_team(tb_a, gk, SD.TRAINEE), SD.draw_team(tb_b, gk, SD.OPPONENT)
        team_a, team_b = (t2, t1) if swapped else (t1, t2)
        sta.team, stb.team = team_a, team_b
        a_is_p1 = (g // 2) % 2 == 0
        p1, p2 = (pa, pb) if a_is_p1 else (pb, pa)
        n1, n2 = (a, b) if a_is_p1 else (b, a)
        p1._botrr_is_p1, p2._botrr_is_p1 = True, False
        _set_streams(p2, SD.bot_stream_seeds(n2, words))
        s1 = SD.bot_route_seed(n1) + P1_ROUTE_OFFSET
        _set_streams(p1, {nm: episode_bot_stream_seed(s1, words, k) for k, nm in enumerate(("choice", "protect"))})
        before = set(pa._battles)
        asyncio.run(run_local_battles(p1, p2, 1, concurrency=1, impl="rust", seed=list(words)))
        tag = next((t for t in pa._battles if t not in before), None)
        bt = pa._battles[tag] if tag is not None else None
        a_p1_games += int(a_is_p1)
        if bt is None or not bt.finished:
            unfinished += 1
            pts.append(None)
            continue
        res, kind = classify_result(won=bt.won, lost=bt.lost, finished=True, turn=bt.turn, turn_cap=limit)
        timeouts += int(kind == "timeout" or (res == DRAW and int(bt.turn or 0) >= limit))
        pts.append(MP.result_points(res))
        bump(L.team_id(team_a), "p", res == WIN)
        bump(L.team_id(team_b), "o", res == LOSS)
        w, l, d = w + (res == WIN), l + (res == LOSS), d + (res == DRAW)
    wall = time.perf_counter() - t0
    pc = MP.pair_counts(pts)
    voided = n_pairs - MP.n_pairs(pc)
    if voided:
        # a voided pair's finished half must leave the counts too: re-derive from the kept pairs only
        raise RuntimeError(f"{key} batch {batch}: {unfinished} unfinished games ({voided} voided pairs) — "
                           "refusing to write a row whose counts and pairs disagree")
    writer = _W["writer"]
    if a not in _W["blocks"]:
        _W["blocks"][a] = bot_block(a)
    if b not in _W["blocks"]:
        _W["blocks"][b] = bot_block(b)
    row = {
        "schema": L.SCHEMA, "row_id": writer.next_row_id(), "supersedes": None, "ts": L.utc_now(),
        "t_start": t_start, "t_end": L.utc_now(), "run": RUN_LABEL, "commit": _W["commit"],
        "player": _W["blocks"][a], "opponent": _W["blocks"][b], "regime": dict(_W["regime"]),
        "compute": {"device": "cpu", "workers_pid": os.getpid(), "wall_s": round(wall, 3),
                    "games_per_s": round(2 * n_pairs / wall, 3) if wall > 0 else None,
                    "player_p1_games": a_p1_games, "timeouts": timeouts, "p1_route_offset": P1_ROUTE_OFFSET,
                    "load_avg": [round(x, 2) for x in os.getloadavg()]},
        "purpose": "anchor",
        "counts": {"w": w, "l": l, "d": d},
        "pairs": {"counts": pc, "n_pairs": n_pairs, "voided": 0},
        "teams": teams,
        "seed": {"rule": SD.SCHEMA, "schedule_seed": int(schedule_seed), "schedule_key": key, "batch": int(batch),
                 "cycle_seed": int(cseed), "item_key": key, "game_lo": 0, "game_hi": 2 * n_pairs - 1},
    }
    writer.append(row)
    return {"edge": key, "batch": batch, "w": w, "l": l, "d": d, "wall": round(wall, 1), "timeouts": timeouts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("play", "plan"))
    ap.add_argument("--pairs", type=int, required=True, help="mirrored pairs per edge (total, across runs)")
    ap.add_argument("--batch-pairs", type=int, default=100)
    ap.add_argument("--schedule-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--edges", default="", help="comma-separated 'a:b' subset (default: all 36)")
    args = ap.parse_args()

    plan = batch_plan(args.pairs, args.batch_pairs)
    reg = regime()
    done = done_units(args.out, reg["regime_id"], args.schedule_seed)
    sel = edges()
    if args.edges:
        want = set(args.edges.split(","))
        sel = [e for e in sel if f"{e[0]}:{e[1]}" in want]
    for (k, bi), n in done.items():
        if n != args.batch_pairs:
            raise SystemExit(f"{args.out} holds {k} batch {bi} with {n} pairs; this plan uses {args.batch_pairs}")
    # interleave edges so a partial run covers every edge evenly
    units = [(a, b, bi, n, args.schedule_seed) for bi, n in enumerate(plan) for (a, b) in sel
             if (item_key(a, b), bi) not in done]
    print(f"[botrr] regime {reg['regime_id']}; {len(sel)} edges x {len(plan)} batches of {args.batch_pairs} pairs; "
          f"{len(done)} units on disk, {len(units)} to play ({2 * args.batch_pairs * len(units)} games)", flush=True)
    if args.cmd == "plan" or not units:
        return 0
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.out,), maxtasksperchild=40) as pool:
        for i, r in enumerate(pool.imap_unordered(play_unit, units), 1):
            el = time.time() - t0
            print(f"[botrr] {i}/{len(units)} {r['edge']} b{r['batch']}: W/L/D {r['w']}/{r['l']}/{r['d']} "
                  f"to {r['timeouts']} in {r['wall']}s; elapsed {el / 60:.1f} min, eta "
                  f"{el / i * (len(units) - i) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
