#!/usr/bin/env python3
"""DOES A TRAINING ROLLOUT ASSIGN THE TRAINEE'S TEAM INDEPENDENTLY OF THE OPPONENT?

The N-curve (`measurements/winprob_refit_ncurve_2026-09-10/` §9) names this as its single largest
open question. Its §3 finding was that in the probe read's EVAL frames the trainee's OWN TEAM
alone predicted the opponent's class at AUC 0.856 / 0.877, because only 37 of 180 and 47 of 216
distinct teams ever faced a sentinel there. If a TRAINING rollout has the same property, the online
head has a turn-1 tell that a matched-team eval frame denies it, and every turn-1 meter on the
critic-ladder campaign carries an own-team mediation term.

This REPLAYS the training-side draw. No GPU, no server, no launcher, no model: the opponent draw is
`MaskableAgentWrapper._select_episode_opponent` against a stub env (the wrapper's own unit tests do
exactly this), and the team draw is the real `Gen3Teambuilder` over the real 719-team pool. The
ORDER is the one `reset()` uses:

    wrappers.py:537-547   reset() -> _select_episode_opponent() -> _apply_opponent_team()
                                  -> super().reset()
    poke_env/environment/env.py:673-675  -> agent1.battle_against(agent2)
    poke_env/player/player.py:759 / 597  -> get_next_team() on EACH side
    poke_env/player/player.py:873-877    -> self._team.yield_team()
    utils/teambuilder.py:192-228         -> yield_team() -> _draw_team()

🚨 THE INTERLEAVING IS MODELLED ON PURPOSE. Both teambuilders default to the `random` MODULE as
their stream (teambuilder.py:65-69), so the trainee's and the opponent's draws are consecutive
calls off ONE Mersenne state, and `random.choice` consumes a number of words that depends on the
sequence LENGTH. A specialist opponent piloting a 1-6 team pin therefore advances the shared stream
by a different amount than the flat 719-team pool does — which means the opponent's class can shift
the PHASE of the stream the next episode's trainee draw comes off. That is an interleaving artefact
and not a dependence, but it is the only channel by which one could exist, so it is reproduced here
rather than assumed away.

Run: `python3 measure.py --out .` (a few minutes, CPU, nothing written under models/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np

from agents.training.snapshot_pool import heuristic_fraction
from agents.training.matchup_spec import MatchupSpec
from agents.training.wrappers import MaskableAgentWrapper
from utils.team_loader.loader import TeamLoader

CLASS_NAME = {MaskableAgentWrapper.OPP_CLASS_BOT: "bot",
              MaskableAgentWrapper.OPP_CLASS_POOL: "pool",
              MaskableAgentWrapper.OPP_CLASS_STABLE: "stable",
              MaskableAgentWrapper.OPP_CLASS_EXPLOITER: "exploiter"}
#: the nine scripted bots a real rollout carries (`_EVAL_OPPONENT_SPECS` is the eval roster; the
#: training roster is built in `main/train/matchup_setup.py` from the same heuristic classes).
N_HEURISTICS = 9
#: how many episodes a pool snapshot identity survives — `_ensure_pool_model` re-samples ONCE per
#: generation, not per episode (wrappers.py:410-430), because loading a snapshot costs ~27 MB.
EPISODES_PER_GENERATION = 500


class _Args:
    """The production ladder arm's argv, reduced to the fields `MatchupSpec.from_args` reads.

    Taken from `models/ai_v12_11_ladder_ctrl10M/metadata.json`'s `original_command`: `--self-play`,
    `--team-pfsp off`, `--team-block-episodes 1`, no `--trainee-team(s)`, no `--stable-opponents`,
    no `--exploiter`. That is the ladder's own opponent mix and the one every arm on the campaign
    was trained under.
    """
    trainee_team = None
    trainee_teams = None
    exploiter = None
    self_play = True
    bot_weights = None
    stable_opponent_temp = 1.0
    team_pfsp = "off"
    team_pfsp_cap = 3.0
    team_pfsp_floor = 0.05
    distill_teacher = None
    distill_team_bias = 0.0


def _stub_wrapper(*, fraction: float, rng_seed: int) -> Tuple[Any, List[Any], Any]:
    env = MagicMock()
    env.agent1.username = "a1"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    heuristics = [MagicMock(name=f"heur{i}") for i in range(N_HEURISTICS)]
    pool = MagicMock()
    pool.is_empty.return_value = False
    pool.load_model.return_value = "MODEL"
    pool_player = MagicMock(name="pool_player")
    w = MaskableAgentWrapper(env, heuristic_opponents=heuristics, pool=pool,
                             pool_player=pool_player, self_play_fraction=fraction,
                             rng_seed=rng_seed)
    return w, heuristics, pool


def simulate(*, n_workers: int, episodes_per_worker: int, fraction: float,
             all_teams: List[str], sample_teams: List[str], global_seed: int) -> Dict[str, Any]:
    """One configuration -> per-episode `(worker, team_key, team_idx, opp_class, opp_id)`."""
    spec = MatchupSpec.from_args(_Args())
    rows: Dict[str, List] = defaultdict(list)
    random.seed(global_seed)                     # the SHARED module stream both builders use
    for widx in range(n_workers):
        # every worker builds its own pair of builders, exactly as a forkserver child does
        trainee_tb = spec.trainee_teams.build(all_teams, sample_teams)
        opponent_tb = spec.opponent_teams.build(all_teams, sample_teams)
        keys = trainee_tb.get_pool_team_keys()
        key_of_packed = {p: k for p, k in zip(trainee_tb.packed_teams, keys)}
        w, heuristics, pool = _stub_wrapper(fraction=fraction, rng_seed=widx)
        heur_id = {id(h): f"bot_{i}" for i, h in enumerate(heuristics)}
        gen = -1
        snap_id = "pool_0"
        for ep in range(episodes_per_worker):
            if ep // EPISODES_PER_GENERATION != gen:
                gen = ep // EPISODES_PER_GENERATION
                # the snapshot identity a generation is pinned to (`_ensure_pool_model`)
                snap_id = f"pool_{widx}_{gen}"
            w._select_episode_opponent()
            w._apply_opponent_team()
            packed = trainee_tb.yield_team()
            idx = trainee_tb._last_pool_idx
            opponent_tb.yield_team()             # the OPPONENT side of the same reset()
            cls = CLASS_NAME[w._opponent_class]
            rows["worker"].append(widx)
            rows["team_key"].append(key_of_packed.get(
                packed, "bias_" + hashlib.sha1(packed.encode()).hexdigest()[:10]))
            rows["team_idx"].append(-1 if idx is None else int(idx))
            rows["opp_class"].append(cls)
            rows["opp_id"].append(snap_id if cls == "pool" else heur_id[id(w.opponent)])
    return {k: np.asarray(v) for k, v in rows.items()}


# --------------------------------------------------------------------------- statistics

def loo_auc(team: np.ndarray, y: np.ndarray) -> float:
    """OWN-TEAM leave-one-episode-out decode of the binary label, scored by AUC.

    The N-curve's own estimator (`frame_check.py`): each episode's feature is the rate of `y`
    among the OTHER episodes that drew the same team, so a team never scores against itself. On an
    independent draw this reads the permutation null; on the probe read's eval frames it read
    0.856 / 0.877.
    """
    teams, inv = np.unique(team, return_inverse=True)
    n = np.bincount(inv, minlength=teams.size).astype(float)
    s = np.bincount(inv, weights=y, minlength=teams.size)
    den = n[inv] - 1.0
    feat = np.where(den > 0, (s[inv] - y) / np.where(den > 0, den, 1.0), np.nan)
    ok = np.isfinite(feat)
    return _auc(y[ok], feat[ok])


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y > 0.5
    if not pos.any() or pos.all():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=float)
    sp = p[order]
    i = 0
    while i < sp.size:
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = float(pos.sum()), float((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """I(a; b) in BITS, plug-in. Biased UPWARD at finite n — read it against the permutation null
    below it, never against zero."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    tab = np.zeros((ai.max() + 1, bi.max() + 1), dtype=float)
    np.add.at(tab, (ai, bi), 1.0)
    n = tab.sum()
    pab = tab / n
    pa = pab.sum(1, keepdims=True)
    pb = pab.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = pab * np.log2(pab / (pa * pb))
    return float(np.nansum(t))


def chi2_table(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Pearson chi-square of independence, with the Monte-Carlo p-value that a 719 x 2 table with
    small expected counts needs (the asymptotic distribution is not to be trusted there)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    tab = np.zeros((ai.max() + 1, bi.max() + 1), dtype=float)
    np.add.at(tab, (ai, bi), 1.0)
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = float(np.nansum(np.where(exp > 0, (tab - exp) ** 2 / np.where(exp > 0, exp, 1), 0)))
    dof = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return {"chi2": stat, "dof": int(dof), "min_expected": float(exp.min()),
            "chi2_over_dof": stat / dof if dof else float("nan")}


def permutation_null(fn, a: np.ndarray, b: np.ndarray, *, n: int, seed: int) -> Dict[str, float]:
    """The reference every number above is read against: `b` shuffled, everything else held.

    🚨 The N-curve's standing rule (§9): *before reading a decode of X from a representation,
    decode X from the nuisance variables the observation carries anyway.* Here the whole point is
    that the plug-in MI of a 719-category variable against a 2-category one is bounded away from
    zero at any finite n, so an unshuffled number quoted against 0 would "detect" independence's
    own sampling noise.
    """
    rng = np.random.default_rng(seed)
    draws = np.asarray([fn(a, rng.permutation(b)) for _ in range(int(n))], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        # 🚨 REFUSE rather than emit a NaN that reads like a null. The one way this happens here
        # is a DEGENERATE frame — a self-play fraction of 0.0 makes every episode a bot episode,
        # so the class label has one level and an AUC is undefined. That is a property of the
        # configuration, not a measurement, and it is said in words.
        return {"refused": "the statistic is undefined on every permutation of this frame — "
                           "the label has a single level (check the class counts above)",
                "n": int(draws.size)}
    return {"mean": float(draws.mean()), "sd": float(draws.std(ddof=1)),
            "p2.5": float(np.percentile(draws, 2.5)),
            "p97.5": float(np.percentile(draws, 97.5)), "n": int(draws.size)}


def ks_two_sample(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Two-sample KS on the TEAM INDEX, pool-drawn episodes against bot-drawn ones."""
    x, y = np.sort(x.astype(float)), np.sort(y.astype(float))
    allv = np.concatenate([x, y])
    cx = np.searchsorted(x, allv, side="right") / x.size
    cy = np.searchsorted(y, allv, side="right") / y.size
    d = float(np.max(np.abs(cx - cy)))
    ne = x.size * y.size / (x.size + y.size)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return {"D": d, "p": float(min(max(p, 0.0), 1.0)), "n_pool": int(x.size), "n_bot": int(y.size)}


def analyse(rows: Dict[str, np.ndarray], *, perm: int, seed: int) -> Dict[str, Any]:
    team, cls, oid, tidx = (rows["team_key"], rows["opp_class"], rows["opp_id"],
                            rows["team_idx"])
    y = (cls == "pool").astype(float)
    out: Dict[str, Any] = {
        "n_episodes": int(team.size),
        "n_distinct_teams": int(np.unique(team).size),
        "n_teams_facing_pool": int(np.unique(team[y > 0.5]).size),
        "n_teams_facing_bot": int(np.unique(team[y < 0.5]).size),
        "pool_share": float(y.mean()),
        "class_counts": {k: int(v) for k, v in Counter(cls.tolist()).items()},
    }
    if np.unique(cls).size < 2:
        out["degenerate"] = ("this configuration draws a SINGLE opponent class, so every "
                             "class-conditional statistic is undefined. Reported as a frame "
                             "census only — never as an independence result.")
        return out
    # (i) own team -> opponent CLASS
    auc = loo_auc(team, y)
    out["i_own_team_to_opp_class"] = {
        "loo_auc": auc,
        "loo_auc_null": permutation_null(lambda a, b: loo_auc(a, b), team, y,
                                         n=perm, seed=seed),
        "mutual_information_bits": mutual_information(team, cls),
        "mutual_information_null": permutation_null(mutual_information, team, cls,
                                                    n=perm, seed=seed + 1),
    }
    # (ii) own team -> opponent IDENTITY
    out["ii_own_team_to_opp_identity"] = {
        "n_distinct_opponents": int(np.unique(oid).size),
        "mutual_information_bits": mutual_information(team, oid),
        "mutual_information_null": permutation_null(mutual_information, team, oid,
                                                    n=perm, seed=seed + 2),
        "entropy_of_opponent_bits": float(-sum(
            (c / oid.size) * math.log2(c / oid.size) for c in Counter(oid.tolist()).values())),
    }
    # (iii) does the TEAM distribution differ by opponent class?
    chi = chi2_table(team, cls)
    chi["mc_null"] = permutation_null(lambda a, b: chi2_table(a, b)["chi2"], team, cls,
                                      n=min(perm, 200), seed=seed + 3)
    pool_idx, bot_idx = tidx[(y > 0.5) & (tidx >= 0)], tidx[(y < 0.5) & (tidx >= 0)]
    out["iii_team_distribution_by_class"] = {
        "chi_square": chi,
        "ks_on_team_index": ks_two_sample(pool_idx, bot_idx),
        "bias_draw_share_pool": float(np.mean(tidx[y > 0.5] < 0)),
        "bias_draw_share_bot": float(np.mean(tidx[y < 0.5] < 0)),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--workers", type=int, default=48)      # the arm's own --n-envs
    ap.add_argument("--episodes", type=int, default=4000)   # per worker
    ap.add_argument("--perm", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--fractions", default="0.0,0.25,0.5,0.75,0.9")
    args = ap.parse_args()

    tl = TeamLoader()
    all_teams, sample_teams = tl.get_all_teams(), tl.get_sample_teams()
    doc: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_all_teams": len(all_teams), "n_sample_teams": len(sample_teams),
        "config": {"workers": args.workers, "episodes_per_worker": args.episodes,
                   "perm": args.perm, "seed": args.seed,
                   "argv_source": "models/ai_v12_11_ladder_ctrl10M/metadata.json "
                                  "original_command (self-play, team-pfsp off, "
                                  "team-block-episodes 1, no trainee-team pin)",
                   "episodes_per_generation": EPISODES_PER_GENERATION,
                   "n_heuristics": N_HEURISTICS},
        "heuristic_fraction_reference": {
            f"{wr:.2f}": heuristic_fraction(wr) for wr in (0.3, 0.5, 0.7, 0.9, 0.99)},
        "arms": {},
    }
    for f in [float(x) for x in args.fractions.split(",")]:
        t0 = time.time()
        rows = simulate(n_workers=args.workers, episodes_per_worker=args.episodes,
                        fraction=f, all_teams=all_teams, sample_teams=sample_teams,
                        global_seed=args.seed + int(f * 1000))
        res = analyse(rows, perm=args.perm, seed=args.seed)
        res["seconds"] = round(time.time() - t0, 1)
        doc["arms"][f"self_play_fraction={f}"] = res
        if "degenerate" in res:
            print(f"f={f}: DEGENERATE — {res['class_counts']}  [{res['seconds']}s]", flush=True)
            continue
        print(f"f={f}: n={res['n_episodes']} teams={res['n_distinct_teams']} "
              f"pool_share={res['pool_share']:.3f} "
              f"LOO-AUC={res['i_own_team_to_opp_class']['loo_auc']:.4f} "
              f"(null {res['i_own_team_to_opp_class']['loo_auc_null']['p2.5']:.4f}"
              f"..{res['i_own_team_to_opp_class']['loo_auc_null']['p97.5']:.4f}) "
              f"chi2/dof={res['iii_team_distribution_by_class']['chi_square']['chi2_over_dof']:.3f}"
              f" KS p={res['iii_team_distribution_by_class']['ks_on_team_index']['p']:.3f}"
              f"  [{res['seconds']}s]", flush=True)
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
