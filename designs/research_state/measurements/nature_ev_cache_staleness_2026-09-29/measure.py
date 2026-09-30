"""How often does Gen3Env serve STALE nature/EV labels? (TECH_DEBT row "Stale nature/EV label cache", Lane C
F-LC-6, step 0 — a MEASUREMENT, no fix.)

THE DEFECT. `Gen3Env._nature_ev_map` caches the opponent team's inverted (species -> nature, EVs) map keyed by
the team's SPECIES SET alone (`frozenset(to_id_str(m.species) for m in b2.team.values())`). The cache lives on
the env instance and survives across battles, so when an env's NEXT opponent team has the SAME six species but a
DIFFERENT spread on any of them, that battle is supervised with the previous team's nature/EV labels.

THE DRAW (production, read from the code, not assumed). Agent2's team comes from `opponent_teambuilder`
= `matchup.opponent_teams.build(...)`, which for the production matchup ("opponent teams: full pool") is
`Gen3Teambuilder(all_teams)` — every opponent kind (bots, self-play snapshots, exploiter targets) draws from it —
with `--team-pfsp off --team-block-episodes 1`: `yield_team()` is ONE uniform `random.choice(packed_teams)` per
battle. Each env worker holds its own builder copy and its own Gen3Env, so a stale label needs two CONSECUTIVE
draws of one env. The per-battle rate is therefore the pairwise rate over the packed pool:
    P(stale) = (1/N^2) * #{(i, j): species_set(i) == species_set(j) and spread(i) != spread(j)}
computed EXACTLY here, and CONFIRMED by driving the real `yield_team()` (seeded) through a long per-env sequence.
The CI is BATCH-MEANS over the independent env streams (a t-interval on the per-env rates), because consecutive
transitions of one stream share a draw and are not independent; in practice it is about as wide as a binomial
(Wilson) interval, which is printed beside it as NAIVE. Under the uniform draw the EXACT value IS the long-run
rate; the simulation checks the draw, not the arithmetic (see README.md for the seed scatter).
The ladder corpus (`utils.ladder_corpus`, drawn by `team_sources.PackedTeamPool`, uniform) is
measured the same way: it is not a training source today (X9), but it is the one the fix must hold on.

"SPREAD" = the per-species (nature, EVs, IVs, level) of the PACKED team (what the sim computes the stats from;
the pool's packed teams carry `fix_gen3_hp_ivs`'s IVs). Two teams differ in spread when any species of the shared
set differs in that tuple. A difference that happens to invert to the same (nature, EVs) is counted as stale
here, so the measured rate is an upper bound on label-visible staleness; the per-species breakdown shows which
fields differ.

    PYTHONPATH=src python designs/research_state/measurements/nature_ev_cache_staleness_2026-09-29/measure.py \
        [--sim-per-env 20000] [--envs 48] [--seed 20260929] [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys

from poke_env.data.normalize import to_id_str

STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def parse_packed(packed: str) -> dict:
    """species id -> (nature, evs, ivs, level) for one PACKED team (Showdown pack format:
    NICK|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS[,…] per mon, ']'-joined)."""
    out = {}
    for mon in packed.split("]"):
        f = mon.split("|")
        if len(f) < 9:
            continue
        species = f[1] or f[0]
        nature = f[5] or "Serious"
        evs = tuple(int(x) if x else 0 for x in (f[6].split(",") if f[6] else [""] * 6))
        ivs = tuple(int(x) if x else 31 for x in (f[8].split(",") if f[8] else [""] * 6))
        level = int(f[10]) if len(f) > 10 and f[10] else 100
        out[to_id_str(species)] = (nature, evs, ivs, level)
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def differing_fields(a: dict, b: dict) -> list:
    """Per shared species: which of nature / evs / ivs / level differ."""
    diffs = []
    for sp in sorted(a):
        if a[sp] != b[sp]:
            fields = [n for n, x, y in zip(("nature", "evs", "ivs", "level"), a[sp], b[sp]) if x != y]
            diffs.append((sp, fields))
    return diffs


def exact(packed: list) -> dict:
    parsed = [parse_packed(p) for p in packed]
    keys = [frozenset(p) for p in parsed]
    n = len(packed)
    groups = collections.defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    stale_pairs = 0
    offenders = []
    field_counts = collections.Counter()
    for k, idx in groups.items():
        if len(idx) < 2:
            continue
        distinct_spreads = {tuple(sorted(parsed[i].items())) for i in idx}
        grp_stale = 0
        for i in idx:
            for j in idx:
                if parsed[i] != parsed[j]:
                    grp_stale += 1
                    for _sp, fields in differing_fields(parsed[i], parsed[j]):
                        field_counts.update(fields)
        stale_pairs += grp_stale
        if grp_stale:
            offenders.append({
                "species_set": sorted(k), "teams_in_pool": len(idx),
                "distinct_spreads": len(distinct_spreads),
                "stale_ordered_pairs": grp_stale,
                "share_of_stale": None,  # filled below
                "example_diff": differing_fields(*[parsed[i] for i in idx
                                                   if parsed[i] != parsed[idx[0]]][:1], parsed[idx[0]])
                if any(parsed[i] != parsed[idx[0]] for i in idx) else [],
            })
    for o in offenders:
        o["share_of_stale"] = o["stale_ordered_pairs"] / stale_pairs if stale_pairs else 0.0
    offenders.sort(key=lambda o: -o["stale_ordered_pairs"])
    same_set_pairs = sum(len(v) ** 2 for v in groups.values())
    return {
        "n_teams": n, "n_species_sets": len(groups),
        "sets_shared_by_2plus_teams": sum(1 for v in groups.values() if len(v) > 1),
        "p_same_set": same_set_pairs / n ** 2,
        "p_stale": stale_pairs / n ** 2,
        "stale_ordered_pairs": stale_pairs,
        "field_counts_over_stale_pair_species": dict(field_counts),
        "offenders": offenders,
    }


#: two-sided 97.5 % t quantiles for the batch-means interval, by degrees of freedom (scipy-free)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262, 19: 2.093, 29: 2.045, 47: 2.012}


def _t975(df: int) -> float:
    return _T975.get(df) or _T975[max(k for k in _T975 if k <= df)]


def simulate(yield_fn_factory, envs: int, per_env: int) -> dict:
    """Drive `envs` independent draw streams of `per_env` battles each through the REAL yield function;
    count consecutive-battle transitions that hit the stale case. CI = batch means over the streams."""
    k = n = 0
    per_env_rates = []
    for e in range(envs):
        yf = yield_fn_factory(e)
        prev = None
        ke = ne = 0
        for _ in range(per_env):
            cur = parse_packed(yf())
            if prev is not None:
                ne += 1
                if frozenset(cur) == frozenset(prev) and cur != prev:
                    ke += 1
            prev = cur
        k += ke; n += ne
        per_env_rates.append(ke / ne)
    m = sum(per_env_rates) / envs
    sd = math.sqrt(sum((r - m) ** 2 for r in per_env_rates) / (envs - 1)) if envs > 1 else float("nan")
    h = _t975(envs - 1) * sd / math.sqrt(envs)
    lo, hi = wilson(k, n)
    return {"transitions": n, "stale": k, "rate": k / n if n else float("nan"),
            "batch_means95": [m - h, m + h], "naive_wilson95": [lo, hi]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-per-env", type=int, default=20000)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = {"method": "exact pairwise over the packed pool + seeded simulation through the real draw",
           "seed": a.seed, "envs": a.envs, "sim_per_env": a.sim_per_env}

    # --- the TRAINING pool, exactly as production builds the opponent side ---
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder
    all_teams = TeamLoader().get_all_teams()
    tb = Gen3Teambuilder(all_teams)          # == matchup.opponent_teams.build for kind "pool"
    pool_packed = list(tb.packed_teams)
    out["pool"] = {"source": "Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams",
                   "loaded_texts": len(all_teams), **exact(pool_packed)}

    def pool_factory(e):
        t = Gen3Teambuilder.__new__(Gen3Teambuilder)
        t.__dict__.update(tb.__dict__)       # a per-env copy of the SAME builder (as each worker unpickles one)
        t._rng = random.Random(a.seed * 1000 + e)
        return t.yield_team
    out["pool"]["simulated"] = simulate(pool_factory, a.envs, a.sim_per_env)

    # --- the LADDER corpus (full tier), drawn by PackedTeamPool ---
    from utils import ladder_corpus
    from utils.team_sources import PackedTeamPool
    lad = ladder_corpus.teams("full")
    out["ladder_full"] = {"source": "utils.ladder_corpus.teams('full') via PackedTeamPool", **exact(lad)}
    out["ladder_full"]["simulated"] = simulate(
        lambda e: PackedTeamPool(lad, rng_seed=a.seed * 1000 + e).yield_team, a.envs, a.sim_per_env)

    for name in ("pool", "ladder_full"):
        r = out[name]
        s = r["simulated"]
        print(f"== {name}: {r['n_teams']} teams, {r['n_species_sets']} species sets "
              f"({r['sets_shared_by_2plus_teams']} shared by 2+ teams)")
        print(f"   EXACT  P(same species set, consecutive draws) = {r['p_same_set']:.5%}")
        print(f"   EXACT  P(stale nature/EV labels per battle)   = {r['p_stale']:.5%}")
        print(f"   SIMULATED ({s['transitions']:,} transitions, real draw): {s['stale']:,} stale = "
              f"{s['rate']:.5%} [batch-means 95 % {s['batch_means95'][0]:.5%}, {s['batch_means95'][1]:.5%}]"
              f"  (naive Wilson {s['naive_wilson95'][0]:.5%}, {s['naive_wilson95'][1]:.5%})")
        print(f"   differing fields over stale (pair, species): {r['field_counts_over_stale_pair_species']}")
        for o in r["offenders"][:8]:
            print(f"   offender {o['share_of_stale']:6.1%} of stale | {o['teams_in_pool']} teams, "
                  f"{o['distinct_spreads']} spreads | {'/'.join(o['species_set'])} | e.g. {o['example_diff']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, default=list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
