"""Power of X72's Kakuna read (PREREG §8), computed BEFORE any game of it, THROUGH THE RULE'S OWN CODE
(`main.b1.kakuna_rule.contrast` / `classify`). A simulation: no game is played and none is read.

The design, as registered: arms `c0` (the eight C0 finals), `plat` (the three read finals) and `s0` (the three
banked swap finals); panels F (forum72), H (heldout72_hpfix) and P (p72); unit seeds 0-59 = 600 deals x 2 games =
1,200 games a final a panel. The generative model and the deal design are
`../x67_entropy_screen_2026-11-09/kakuna_power.py` part 2's (BOUNDARY-1's ASSUMED team model, and the same with no
team-by-arm interaction), with a third panel drawn like the other two. **P72's team SDs are ASSUMED equal to the
other panels'**: no banked P72 game of these finals exists to measure them (PREREG §8.6).

Targets:
    plat_U   plat - c0 on F, H and P, equal weight      THE DECIDING contrast (3 finals against 8)
    plat_FH  plat - c0 on F and H                       reported (the banked +2.61 pp's frame)
    dose_U   plat - s0 on F, H and P                    reported (3 finals against 3)
In the simulation `plat` carries the effect and `s0` sits at the control's level, so the true value of every target
is the effect. A final's seed shift is shared by its panels.

Deterministic: `SeedSequence([20261125, model index, sigma index, effect index, chunk])`. CPU only, from the repo
root with PYTHONPATH=src (the optional argument is the worker count; the output does not depend on it):
    python designs/research_state/measurements/x72_swap_plateau_2026-11-25/kakuna_power.py 5 > kakuna_power.log
"""
import math
import random
import sys
from multiprocessing import Pool

import numpy as np

from main.b1.kakuna_rule import NEGATIVE, POSITIVE, classify, contrast

SEED = 20261125
LEVEL = 0.90
TEAMS, UNIT_SEEDS, DEALS = 72, 60, 10
BASE = 0.51
ARMS = {"c0": [f"c{i}" for i in range(8)], "plat": ["p0", "p1", "p2"], "s0": ["s0", "s1", "s2"]}
PANELS = ("F", "H", "P")
THIRD = 1.0 / 3.0
TARGETS = {
    "plat_U": ({"plat": 1.0, "c0": -1.0}, {"F": THIRD, "H": THIRD, "P": THIRD}),
    "plat_FH": ({"plat": 1.0, "c0": -1.0}, {"F": 0.5, "H": 0.5}),
    "dose_U": ({"plat": 1.0, "s0": -1.0}, {"F": THIRD, "H": THIRD, "P": THIRD}),
}
SIGMAS = (0.0, 0.5, 1.0, 1.5)
EFFECTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
REPS, CHUNK = 2000, 250


def design():
    def draws(s):
        r = random.Random(s)
        return [r.choice(range(TEAMS)) for _ in range(DEALS)]

    a = np.array([t for s in range(UNIT_SEEDS) for t in draws(s)])
    b = np.array([t for s in range(UNIT_SEEDS) for t in draws(s + 1)])
    return a, b


A, B = design()
TWO = np.full(len(A), 2.0)


def experiment(rng, sigma, effect, m):
    seed = {r: sigma / 100 * rng.standard_normal() for labs in ARMS.values() for r in labs}
    data = {r: {} for r in seed}
    for p in PANELS:
        our = m["sd_our"] / 100 * rng.standard_normal(TEAMS)
        their = m["sd_their"] / 100 * rng.standard_normal(TEAMS)
        for arm, labs in ARMS.items():
            inter = m["tau"] / 100 * rng.standard_normal(TEAMS)
            for r in labs:
                prob = BASE + (effect / 100 if arm == "plat" else 0.0) + seed[r] + our[A] + their[B] + inter[A]
                data[r][p] = (A, B, TWO, rng.binomial(2, np.clip(prob, 0.02, 0.98)).astype(float))
    return data


def chunk(task):
    mi, si, ei, ci, sigma, effect, model, n = task
    rng = np.random.default_rng(np.random.SeedSequence([SEED, mi, si, ei, ci]))
    rows = []
    for _ in range(n):
        data = experiment(rng, sigma, effect, model)
        for name, (coef, pw) in TARGETS.items():
            st = contrast(data, ARMS, coef, pw, LEVEL)
            lo, hi = 100 * st["lo"], 100 * st["hi"]
            cls = classify(lo, hi)["class"]
            rows.append((name, cls == POSITIVE, cls == NEGATIVE, lo <= effect <= hi, (hi - lo) / 2))
    return (mi, si, ei), rows


def part(pool, mi, title, model):
    todo = [(mi, si, ei, ci, sigma, effect, model, min(CHUNK, REPS - ci * CHUNK))
            for si, sigma in enumerate(SIGMAS) for ei, effect in enumerate(EFFECTS)
            for ci in range(math.ceil(REPS / CHUNK))]
    out = {}
    for key, rows in pool.imap_unordered(chunk, todo):
        out.setdefault(key, []).extend(rows)
    print(f"\n{title}: our-team SD {model['sd_our']:g} pp, their-team SD {model['sd_their']:g} pp, team-by-arm "
          f"interaction SD {model['tau']:g} pp; {2 * len(A)} games a final a panel, level {LEVEL}; "
          f"{REPS} experiments a cell (a rate's Monte-Carlo SE at most {0.5 / math.sqrt(REPS):.3f})")
    for name in TARGETS:
        for si, sigma in enumerate(SIGMAS):
            cs = []
            for ei in range(len(EFFECTS)):
                r = [x for x in out[(mi, si, ei)] if x[0] == name]
                cs.append({"pos": np.mean([x[1] for x in r]), "neg": np.mean([x[2] for x in r]),
                           "cover": np.mean([x[3] for x in r]), "half": float(np.median([x[4] for x in r]))})
            z = cs[0]
            power = " ".join(f"+{e:g}: {c['pos']:.2f}" for e, c in zip(EFFECTS[1:], cs[1:]))
            print(f"  {name:<7} seed SD {sigma:.1f} pp | at 0: false POSITIVE {z['pos']:.3f} false NEGATIVE {z['neg']:.3f} "
                  f"coverage {z['cover']:.3f} | median half-width {z['half']:.2f} pp | P(POSITIVE) {power}")


def main():
    a_use = np.bincount(A, minlength=TEAMS)
    print(f"design: unit seeds 0-{UNIT_SEEDS - 1}, {len(A)} deals x 2 games = {2 * len(A)} games a final a panel, "
          f"panels {', '.join(PANELS)}; teams piloted {int((a_use > 0).sum())} of {TEAMS} (uses "
          f"{int(a_use[a_use > 0].min())}-{int(a_use.max())}); arms "
          + ", ".join(f"{k} x{len(v)}" for k, v in ARMS.items()) + f"; seed {SEED}")
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as pool:
        part(pool, 0, "BOUNDARY-1's ASSUMED team model", {"sd_our": 10.0, "sd_their": 9.0, "tau": 5.0})
        part(pool, 1, "no team-by-arm interaction", {"sd_our": 10.0, "sd_their": 9.0, "tau": 0.0})


if __name__ == "__main__":
    main()
