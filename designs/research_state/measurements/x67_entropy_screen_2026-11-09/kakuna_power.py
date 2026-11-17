"""Power of the Kakuna read of the entropy finals (X67 x3 and CE-3d x3 against C0 x8), computed BEFORE any game of it,
THROUGH THE RULE'S OWN CODE (`main.b1.kakuna_rule.contrast` / `classify` / `unit_deals`). No game is played here and no
X67 or CE-3d game is read: part 1 reads only C0's BANKED forum72 games (X60-T look 2's state, already read), part 2
is a simulation.

PART 1, THE BANKED SPREAD. C0's eight finals on forum72, 1,200 games each (unit seeds 0-59), engine E3
(`~/gen3ai_archive/x60t_read/look2/cells.json`, groups primary / primary_c0a / primary_c0b). Every one of the 56 ways to
call three of the eight "the arm" and the other five "the control" is a PLACEBO contrast whose true value is 0: the
registered interval's half-width on real games, and how often it calls a placebo POSITIVE or NEGATIVE. (The splits
share games, so the 56 are not independent; the read itself is three against EIGHT on two panels, which is narrower.)

PART 2, THE REGISTERED DESIGN, simulated as `boundary1_scratch_2026-11-09/kakuna_oc.py` does (same deal design, same
generative model): arms c0 (8 runs) and x (3 runs), panels F and H, unit seeds 0-59 = 600 deals x 2 games = 1,200
games a run a panel. Two team models: BOUNDARY-1's ASSUMED one, and the same with no team-by-arm interaction (part 1
measures the two team SDs on real games, to set beside the assumed ones). Targets: U (F and H, equal weight; the
CONFIRMING contrast) and F alone (a REPORTED one). A run's seed shift is shared by both panels.

Deterministic: `SeedSequence([20261116, model index, sigma index, effect index, chunk])`. CPU only, from the repo
root with PYTHONPATH=src (the optional argument is the worker count; the output does not depend on it):
    python designs/research_state/measurements/x67_entropy_screen_2026-11-09/kakuna_power.py 5 > kakuna_power.log
"""
import itertools
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from main.b1.kakuna_rule import NEGATIVE, POSITIVE, classify, contrast, unit_deals

SEED = 20261116
LEVEL = 0.90
TEAMS, UNIT_SEEDS, DEALS = 72, 60, 10
BASE = 0.51
ARMS = {"c0": [f"c{i}" for i in range(8)], "x": ["x0", "x1", "x2"]}
COEF = {"x": 1.0, "c0": -1.0}
PANELS = ("F", "H")
TARGETS = {"U": {"F": 0.5, "H": 0.5}, "F": {"F": 1.0}}
SIGMAS = (0.0, 0.5, 1.0, 1.5)
EFFECTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
REPS, CHUNK = 2000, 250
BANK = Path.home() / "gen3ai_archive/x60t_read/look2/cells.json"
BANK_GROUPS = ("primary", "primary_c0a", "primary_c0b")


def design():
    def draws(s):
        r = random.Random(s)
        return [r.choice(range(TEAMS)) for _ in range(DEALS)]

    a = np.array([t for s in range(UNIT_SEEDS) for t in draws(s)])
    b = np.array([t for s in range(UNIT_SEEDS) for t in draws(s + 1)])
    return a, b


A, B = design()
TWO = np.full(len(A), 2.0)


# ------------------------------------------------------------------------------------------------ part 1
def banked():
    """``{run: {"forum72": Cell}}`` of C0's eight finals, and the per-run (wins, games)."""
    cells = json.loads(BANK.read_text())["anchor"]
    agg, names = {}, set()
    for g in BANK_GROUPS:
        for u in cells[g]["units"]:
            if not (u["label"].startswith("c0_") and u["panel"] == "forum72" and u["counted"]):
                continue
            by = agg.setdefault(u["label"], {})
            for _half, _index, our, their, win, _tie, _cap in unit_deals(u, {}):
                c = by.setdefault((our, their), [0, 0])
                c[0] += 1
                c[1] += int(win)
                names.update((our, their))
    pos = {t: i for i, t in enumerate(sorted(names))}
    data = {}
    for lab, by in sorted(agg.items()):
        rows = sorted((pos[o], pos[t], n, w) for (o, t), (n, w) in by.items())
        arr = np.array(rows, dtype=float)
        data[lab] = {"forum72": (arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3])}
    return data, len(pos)


def part1():
    data, n_teams = banked()
    runs = sorted(data)
    n = {r: float(data[r]["forum72"][2].sum()) for r in runs}
    w = {r: float(data[r]["forum72"][3].sum()) for r in runs}
    rates = np.array([w[r] / n[r] for r in runs])
    pooled = sum(w.values()) / sum(n.values())
    binom = 100 * math.sqrt(pooled * (1 - pooled) / 1200)
    print(f"PART 1: C0's banked forum72 games (E3): {len(runs)} runs, {int(sum(w.values()))} / {int(sum(n.values()))} = "
          f"{pooled:.4f}; {n_teams} teams in either role")
    print("  per run: " + ", ".join(f"{r} {int(w[r])}/{int(n[r])} = {w[r] / n[r]:.4f}" for r in runs))
    print(f"  SD of the eight run rates {100 * rates.std(ddof=1):.2f} pp; the binomial SD of one run's rate at 1,200 "
          f"games {binom:.2f} pp")
    # team sizes, by our team and by their team (moment estimates net of the binomial part), for part 2's model
    for role, col in (("our", 0), ("their", 1)):
        tot, win = np.zeros(n_teams), np.zeros(n_teams)
        for r in runs:
            c = data[r]["forum72"]
            np.add.at(tot, c[col], c[2])
            np.add.at(win, c[col], c[3])
        m = tot > 0
        tr = win[m] / tot[m]
        noise = float(np.mean(pooled * (1 - pooled) / tot[m]))
        sd = math.sqrt(max(0.0, float(tr.var(ddof=1)) - noise))
        print(f"  {role}-team SD of the win rate, net of game noise: {100 * sd:.1f} pp over {int(m.sum())} teams "
              f"({int(tot[m].min())}-{int(tot[m].max())} games a team)")
    rows = []
    for arm in itertools.combinations(runs, 3):
        arms = {"x": list(arm), "c0": [r for r in runs if r not in arm]}
        st = contrast(data, arms, COEF, {"forum72": 1.0}, LEVEL)
        dec = st["schemes"][st["deciding"]]
        rows.append((100 * st["psi"], 100 * dec["half"], classify(100 * st["lo"], 100 * st["hi"])["class"],
                     st["deciding"], 1e4 * dec["v_s"], 1e4 * dec["v_t"], 1e4 * dec["v_ts"], dec["df"]))
    psi = np.array([r[0] for r in rows])
    half = np.array([r[1] for r in rows])
    print(f"  the 56 placebo contrasts (3 runs - 5 runs, forum72 alone, level {LEVEL}): half-width median "
          f"{np.median(half):.2f} pp (min {half.min():.2f}, max {half.max():.2f}); |psi| median "
          f"{np.median(np.abs(psi)):.2f} pp, max {np.abs(psi).max():.2f} pp; SD of psi {psi.std(ddof=1):.2f} pp")
    print(f"  called POSITIVE {sum(r[2] == POSITIVE for r in rows)} / 56, NEGATIVE {sum(r[2] == NEGATIVE for r in rows)} "
          f"/ 56; dyadic decides in {sum(r[3] == 'dyadic' for r in rows)} / 56; medians V_S "
          f"{np.median([r[4] for r in rows]):.2f} V_T {np.median([r[5] for r in rows]):.2f} V_TS "
          f"{np.median([r[6] for r in rows]):.2f} pp^2, df {np.median([r[7] for r in rows]):.1f}")


# ------------------------------------------------------------------------------------------------ part 2
def experiment(rng, sigma, effect, m):
    seed = {r: sigma / 100 * rng.standard_normal() for labs in ARMS.values() for r in labs}
    data = {r: {} for r in seed}
    for p in PANELS:
        our = m["sd_our"] / 100 * rng.standard_normal(TEAMS)
        their = m["sd_their"] / 100 * rng.standard_normal(TEAMS)
        for arm, labs in ARMS.items():
            inter = m["tau"] / 100 * rng.standard_normal(TEAMS)
            for r in labs:
                prob = BASE + (effect / 100 if arm == "x" else 0.0) + seed[r] + our[A] + their[B] + inter[A]
                data[r][p] = (A, B, TWO, rng.binomial(2, np.clip(prob, 0.02, 0.98)).astype(float))
    return data


def chunk(task):
    mi, si, ei, ci, sigma, effect, model, n = task
    rng = np.random.default_rng(np.random.SeedSequence([SEED, mi, si, ei, ci]))
    rows = []
    for _ in range(n):
        data = experiment(rng, sigma, effect, model)
        for name, pw in TARGETS.items():
            st = contrast(data, ARMS, COEF, pw, LEVEL)
            lo, hi = 100 * st["lo"], 100 * st["hi"]
            cls = classify(lo, hi)["class"]
            rows.append((name, cls == POSITIVE, cls == NEGATIVE, lo <= effect <= hi, (hi - lo) / 2))
    return (mi, si, ei), rows


def part2(pool, mi, title, model):
    todo = [(mi, si, ei, ci, sigma, effect, model, min(CHUNK, REPS - ci * CHUNK))
            for si, sigma in enumerate(SIGMAS) for ei, effect in enumerate(EFFECTS)
            for ci in range(math.ceil(REPS / CHUNK))]
    out = {}
    for key, rows in pool.imap_unordered(chunk, todo):
        out.setdefault(key, []).extend(rows)
    print(f"\nPART 2 ({title}): our-team SD {model['sd_our']:g} pp, their-team SD {model['sd_their']:g} pp, team-by-arm "
          f"interaction SD {model['tau']:g} pp; 3 runs against 8, {2 * len(A)} games a run a panel, level {LEVEL}; "
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
            print(f"  {name} seed SD {sigma:.1f} pp | at 0: false POSITIVE {z['pos']:.3f} false NEGATIVE {z['neg']:.3f} "
                  f"coverage {z['cover']:.3f} | median half-width {z['half']:.2f} pp | P(POSITIVE) {power}")


def main():
    a_use = np.bincount(A, minlength=TEAMS)
    print(f"design: unit seeds 0-{UNIT_SEEDS - 1}, {len(A)} deals x 2 games = {2 * len(A)} games a run a panel; teams "
          f"piloted {int((a_use > 0).sum())} of {TEAMS} (uses {int(a_use[a_use > 0].min())}-{int(a_use.max())}); "
          f"seed {SEED}\n")
    part1()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as pool:
        part2(pool, 0, "BOUNDARY-1's ASSUMED team model", {"sd_our": 10.0, "sd_their": 9.0, "tau": 5.0})
        part2(pool, 1, "no team-by-arm interaction", {"sd_our": 10.0, "sd_their": 9.0, "tau": 0.0})


if __name__ == "__main__":
    main()
