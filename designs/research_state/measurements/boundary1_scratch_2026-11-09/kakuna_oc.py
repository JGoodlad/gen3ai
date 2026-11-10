"""Operating characteristics of BOUNDARY-1's Kakuna meter (rule `b1_kakuna`), simulated THROUGH THE RULE'S OWN CODE:
`contrast`, `classify`, `t_quantile`, `bootstrap_rates` and `bootstrap_interval` are imported from
`main.b1.kakuna_rule` (a test asserts it), so the coverage below is the registered interval's. One `contrast`
call serves both levels: the second level's deciding half-width is the larger of `t_quantile(level, df) * sqrt(V)`
over the two schemes, which is what `contrast` returns at that level (a test asserts that too).

THE DESIGN is the anchor lane's: 8 runs (2 seeds x 4 arms C / W / Br / WB), unit seeds 0-35, 20 games a unit, so 720
games a run a panel. A unit seed s deals 10 (our team, their team) pairs and plays each twice (the two challenge
halves); our team is draw i of `random.Random(s)` over the panel's 72 sorted files and Kakuna's is draw i of
`random.Random(s + 1)` (`main.anchors.peer_scripts.metamon_side`), uniform WITH replacement, so 70 of 72 teams are
piloted, between 1 and 11 times each, and every run plays the same deals. The same index sequence on every panel.

THE GENERATIVE MODEL, per panel and experiment (a linear probability, clipped to [0.02, 0.98]):
P(win) = BASE + effect[arm] + seed[run] + our[team] + their[team] + inter[arm][our team]
(+ inter2[arm][their team] in sensitivity S1), our ~ N(0, sd_our), their ~ N(0, sd_their), inter ~ N(0, tau) drawn
fresh per arm and panel, seed[run] ~ N(0, sigma) ONE shift a run shared by every panel (the conservative reading of a
between-seed SD), games Binomial(2, P) a deal. The default sd_our / sd_their are the clustering addendum's design
effects (our team ~10 pp, their team ~9 pp); tau is the draft's team-by-arm interaction (`power.py`, 5 pp). The effect
is added to the two WIDE arms on every panel, so the contrast is the factorial's main effect (width's and breadth's
are the same problem, so target U is `width_U` / `breadth_U` and target F is `breadth_F` / `width_F`): on U (three
panels), on F alone (one panel), and a generic two-panel NET (F plus a fourth panel; its true value is TWICE the
per-panel effect; BOUNDARY-1 registers no net, the row is the rule's net feature).

Deterministic: `SeedSequence([20261109, block, sigma index, effect index, chunk])`. CPU only:
    python kakuna_oc.py > kakuna_oc.log
"""
import argparse
import math
import random
import sys
from multiprocessing import Pool

import numpy as np

from main.b1.kakuna_rule import (NEGATIVE, POSITIVE, bootstrap_interval, bootstrap_rates, classify, contrast,
                                        linear_terms, t_quantile)

SEED = 20261109
#: The level `read_spec_kakuna.json` registers (a test holds the two equal).
REGISTERED_LEVEL = 0.90
TEAMS, UNIT_SEEDS, DEALS = 72, 36, 10
BASE = 0.50
MODEL = {"sd_our": 10.0, "sd_their": 9.0, "tau": 5.0, "tau_their": 0.0}                    # pp
ARMS = {"C": ["c1", "c2"], "W": ["w1", "w2"], "Br": ["b1", "b2"], "WB": ["wb1", "wb2"]}
MAIN = {"W": 0.5, "WB": 0.5, "C": -0.5, "Br": -0.5}
ON = ("W", "WB")
PANELS = ("H", "F", "P72", "X")
TARGETS = {"U": {"H": 1 / 3, "F": 1 / 3, "P72": 1 / 3}, "F": {"F": 1.0}, "NET": {"F": 1.0, "X": 1.0}}
TRUE = {"U": 1.0, "F": 1.0, "NET": 2.0}                      # the target's true value per unit of per-panel effect
CONFIRMING = {"width_U": "U", "breadth_F": "F"}
LEVELS = (0.95, 0.90)
SIGMAS = (0.0, 1.0, 1.5, 2.5)
EFFECTS = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SENSITIVITY = (("S1: a their-team-by-arm interaction of 5 pp too", {"tau_their": 5.0}),
               ("S2: the team-by-arm interaction at 10 pp", {"tau": 10.0}),
               ("S3: no team-by-arm interaction", {"tau": 0.0}),
               ("S4: team main effects of 3 pp (not 10 / 9), interaction 5 pp", {"sd_our": 3.0, "sd_their": 3.0}),
               ("S5: team main effects of 3 pp, interaction 10 pp", {"sd_our": 3.0, "sd_their": 3.0, "tau": 10.0}))
CHUNK = 250


def design():
    """The lane's deals: (our team, their team) index arrays, one row a deal (played twice)."""
    def draws(s):
        r = random.Random(s)
        return [r.choice(range(TEAMS)) for _ in range(DEALS)]

    a = np.array([t for s in range(UNIT_SEEDS) for t in draws(s)])
    b = np.array([t for s in range(UNIT_SEEDS) for t in draws(s + 1)])
    return a, b


A, B = design()
TWO = np.full(len(A), 2.0)


def experiment(rng, sigma, effect, model=MODEL):
    """One simulated read: ``data[run][panel] = (our team, their team, games, wins)`` (every size in pp)."""
    m = dict(MODEL, **model)
    seed = {r: sigma / 100 * rng.standard_normal() for labs in ARMS.values() for r in labs}
    data = {r: {} for r in seed}
    for p in PANELS:
        our = m["sd_our"] / 100 * rng.standard_normal(TEAMS)
        their = m["sd_their"] / 100 * rng.standard_normal(TEAMS)
        for arm, labs in ARMS.items():
            inter = m["tau"] / 100 * rng.standard_normal(TEAMS)
            inter2 = m["tau_their"] / 100 * rng.standard_normal(TEAMS)
            for r in labs:
                prob = BASE + (effect / 100 if arm in ON else 0.0) + seed[r] + our[A] + their[B] + inter[A] + inter2[B]
                data[r][p] = (A, B, TWO, rng.binomial(2, np.clip(prob, 0.02, 0.98)).astype(float))
    return data


def paired_half(data, st, pw, level):
    """FOR COMPARISON, NOT THE REGISTERED INTERVAL: V = V_S + max(0, V_T - V_deal), V_deal the sandwich with one
    cluster a (run, deal) (the game noise V_S already holds), Satterthwaite on (df_S, df_T), the wider scheme."""
    psi = linear_terms(data, st["run_weights"], pw)
    v_deal = sum(float((x ** 2).sum()) * len(x) / (len(x) - 1) for by in psi.values() for x in by.values())
    best = 0.0
    for x in st["schemes"].values():
        b = max(0.0, x["v_t"] - v_deal)
        v = x["v_s"] + b
        df = v * v / (x["v_s"] ** 2 / st["df_s"] + b * b / x["df_t"]) if v > 0 else st["df_s"]
        best = max(best, t_quantile(level, max(df, st["df_s"])) * math.sqrt(v))
    return best


def deciding_half(st, level):
    """The deciding (wider scheme's) half-width of a computed contrast at another level."""
    return max(t_quantile(level, x["df"]) * math.sqrt(x["v"]) for x in st["schemes"].values())


def chunk(task):
    block, si, ei, ci, sigma, effect, model, n, boot = task
    rng = np.random.default_rng(np.random.SeedSequence([SEED, block, si, ei, ci]))
    rows = []
    for _ in range(n):
        data = experiment(rng, sigma, effect, model)
        b = bootstrap_rates(data, ARMS, PANELS, boot, int(rng.integers(0, 2 ** 31))) if boot else None
        for name, pw in TARGETS.items():
            st = contrast(data, ARMS, MAIN, pw, LEVELS[0])
            truth = TRUE[name] * effect
            ours = st["schemes"]["our_team"]
            for level in LEVELS:
                half = deciding_half(st, level)
                lo, hi = 100 * (st["psi"] - half), 100 * (st["psi"] + half)
                cls = classify(lo, hi)["class"]
                alt = 100 * paired_half(data, st, pw, level)
                row = {"name": name, "level": level, "pos": cls == POSITIVE, "neg": cls == NEGATIVE,
                       "cover": lo <= truth <= hi, "half": 100 * half, "dyadic": st["deciding"] == "dyadic",
                       "df": st["schemes"][st["deciding"]]["df"], "v_s": 1e4 * ours["v_s"], "v_t": 1e4 * ours["v_t"],
                       "v_ts": 1e4 * ours["v_ts"], "b_pos": ours["v_t"] > ours["v_ts"], "psi": 100 * st["psi"],
                       "alt_pos": classify(100 * st["psi"] - alt, 100 * st["psi"] + alt)["class"] == POSITIVE,
                       "alt_cover": abs(100 * st["psi"] - truth) <= alt, "alt_half": alt}
                if b is not None:
                    bi = bootstrap_interval(b, ARMS, MAIN, pw, level)
                    row.update({k: 100 * bi[k]["lo"] <= truth <= 100 * bi[k]["hi"] for k in ("literal", "rao_wu")})
                rows.append(row)
    return (block, si, ei), rows


def tasks(block, sigmas, effects, reps, model=MODEL, boot=0):
    return [(block, si, ei, ci, sigma, effect, model, min(CHUNK, reps - ci * CHUNK), boot)
            for si, sigma in enumerate(sigmas) for ei, effect in enumerate(effects)
            for ci in range(math.ceil(reps / CHUNK))]


def run(pool, todo):
    out = {}
    for key, rows in pool.imap_unordered(chunk, todo):
        out.setdefault(key, []).extend(rows)
    return out


def cell(rows, name, level):
    r = [x for x in rows if x["name"] == name and x["level"] == level]
    col = lambda k: np.array([x[k] for x in r], dtype=float)
    out = {k: float(col(k).mean()) for k in ("pos", "neg", "cover", "dyadic", "v_s", "v_t", "v_ts", "b_pos", "alt_pos",
                                             "alt_cover")}
    out.update(n=len(r), half=float(np.median(col("half"))), df=float(np.median(col("df"))),
               alt_half=float(np.median(col("alt_half"))), sd=float(col("psi").std(ddof=1)),
               boot=[float(col("literal").mean()), float(col("rao_wu").mean())] if "literal" in r[0] else None)
    return out


def detect(effects, power, scale):
    """The smallest true value detected with probability 0.80, by linear interpolation on the effect grid."""
    for (e0, p0), (e1, p1) in zip(zip(effects, power), zip(effects[1:], power[1:])):
        if p0 < 0.80 <= p1:
            return f"{scale * (e0 + (e1 - e0) * (0.80 - p0) / (p1 - p0)):.1f}"
    return f"> {scale * effects[-1]:.0f}" if power[-1] < 0.80 else f"<= {scale * effects[0]:.0f}"


def main(reps, boot_reps, boot_draws, procs):
    a_use = np.bincount(A, minlength=TEAMS)
    print(f"design: {len(A)} deals x 2 games = {2 * len(A)} games a run a panel; teams piloted {int((a_use > 0).sum())} "
          f"of {TEAMS} (uses {int(a_use[a_use > 0].min())}-{int(a_use.max())}); teams in either role "
          f"{len(set(A) | set(B))}")
    print(f"model: base {BASE}, our-team SD {MODEL['sd_our']:g} pp, their-team SD {MODEL['sd_their']:g} pp, team-by-arm "
          f"interaction SD {MODEL['tau']:g} pp; seed {SEED}; {reps} experiments a cell (a rate's Monte-Carlo SE is at "
          f"most {0.5 / math.sqrt(reps):.4f})")
    with Pool(procs) as pool:
        res = run(pool, tasks(0, SIGMAS, EFFECTS, reps))
        cells = {(name, level, si): [cell(res[(0, si, ei)], name, level) for ei in range(len(EFFECTS))]
                 for name in TARGETS for level in LEVELS for si in range(len(SIGMAS))}
        print("\n(1) THE REGISTERED ANALYTIC INTERVAL (the wider of our_team / dyadic). Per cell: P(POSITIVE) at each "
              "true per-panel effect; at a true 0, false POSITIVE / false NEGATIVE / coverage.")
        for name in TARGETS:
            for level in LEVELS:
                print(f"\n  target {name} (true value = {TRUE[name]:g} x the per-panel effect), level {level}")
                for si, sigma in enumerate(SIGMAS):
                    cs = cells[(name, level, si)]
                    z = cs[0]
                    power = " ".join(f"+{e:g}: {c['pos']:.3f}" for e, c in zip(EFFECTS[1:], cs[1:]))
                    print(f"    seed SD {sigma:.1f} pp | at 0: false POS {z['pos']:.4f} false NEG {z['neg']:.4f} coverage "
                          f"{z['cover']:.4f} | median half-width {z['half']:.2f} pp | P(POSITIVE) {power} | 0.80-detected "
                          f"true value {detect(EFFECTS, [c['pos'] for c in cs], TRUE[name])} pp")
                    if level == LEVELS[0]:
                        print(f"        at 0: true SD of psi {z['sd']:.2f} pp; mean V_S {z['v_s']:.2f} V_T {z['v_t']:.2f} "
                              f"V_TS {z['v_ts']:.2f} pp^2 (our_team); V_T > V_TS in {z['b_pos']:.3f}; dyadic decides in "
                              f"{z['dyadic']:.3f}; median df {z['df']:.1f}; coverage at +4: {cs[3]['cover']:.4f}")
        print("\n(1b) FOR COMPARISON, NOT REGISTERED: the PAIRED composition V_S + max(0, V_T - V_deal) (the registered "
              "one floors V_S at V_TS, which holds the team main effects that cancel in a contrast of runs playing "
              "the same deals)")
        for name in TARGETS:
            for level in LEVELS:
                for si, sigma in enumerate(SIGMAS):
                    cs = cells[(name, level, si)]
                    power = " ".join(f"+{e:g}: {c['alt_pos']:.3f}" for e, c in zip(EFFECTS[1:], cs[1:]))
                    print(f"    {name} level {level} seed SD {sigma:.1f} | at 0: false POS {cs[0]['alt_pos']:.4f} coverage "
                          f"{cs[0]['alt_cover']:.4f} | median half-width {cs[0]['alt_half']:.2f} pp | P(POSITIVE) {power} | "
                          f"0.80-detected {detect(EFFECTS, [c['alt_pos'] for c in cs], TRUE[name])} pp")
        print("\n(2) SENSITIVITY of the registered interval's coverage at a true 0 (seed SD 0 and 1.5 pp), level 0.95 / "
              "0.90; in brackets the true SD of psi, the 0.95 median half-width, and the paired comparison's coverage")
        for bi, (label, model) in enumerate(SENSITIVITY, start=1):
            r = run(pool, tasks(bi, (0.0, 1.5), (0.0,), reps, model))
            for si, sigma in enumerate((0.0, 1.5)):
                row = []
                for name in TARGETS:
                    c95, c90 = (cell(r[(bi, si, 0)], name, lv) for lv in LEVELS)
                    row.append(f"{name} {c95['cover']:.4f} / {c90['cover']:.4f} (true SD {c95['sd']:.2f}, half "
                               f"{c95['half']:.2f}; paired {c95['alt_cover']:.4f} / {c90['alt_cover']:.4f})")
                print(f"  {label}; seed SD {sigma:.1f} | " + " | ".join(row))
        print(f"\n(3) THE TWO REPORTED BOOTSTRAPS, coverage at a true 0 ({boot_reps} experiments a cell, {boot_draws} "
              "draws each, not the read's 10,000): literal (seeds n-of-n) / Rao-Wu (n - 1 seed draws), level 0.95 then "
              "0.90")
        r = run(pool, tasks(9, SIGMAS, (0.0,), boot_reps, boot=boot_draws))
        for si, sigma in enumerate(SIGMAS):
            row = []
            for name in TARGETS:
                c95, c90 = (cell(r[(9, si, 0)], name, lv) for lv in LEVELS)
                row.append(f"{name} {c95['boot'][0]:.3f} / {c95['boot'][1]:.3f} ; {c90['boot'][0]:.3f} / "
                           f"{c90['boot'][1]:.3f}")
            print(f"  seed SD {sigma:.1f} | " + " | ".join(row))
    print(f"\n(4) THE CONFIRMING CONTRASTS at 720 games a run a panel. REGISTERED LEVEL {REGISTERED_LEVEL} (two-sided; "
          "the other level beside it)")
    for cname, target in CONFIRMING.items():
        for level in sorted(LEVELS, key=lambda lv: lv != REGISTERED_LEVEL):
            tag = "REGISTERED" if level == REGISTERED_LEVEL else "not registered"
            for si, sigma in enumerate(SIGMAS):
                cs = cells[(target, level, si)]
                power = " ".join(f"+{e:g}: {c['pos']:.3f}" for e, c in zip(EFFECTS[1:6], cs[1:6]))
                print(f"  {cname} level {level} ({tag}) seed SD {sigma:.1f} pp | true 0: false POSITIVE {cs[0]['pos']:.4f} "
                      f"false NEGATIVE {cs[0]['neg']:.4f} | P(POSITIVE) {power} | median half-width {cs[0]['half']:.2f} pp "
                      f"| 0.80-detected {detect(EFFECTS, [c['pos'] for c in cs], 1.0)} pp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--boot-draws", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    sys.exit(main(args.reps, args.boot_reps, args.boot_draws, args.procs))
