"""STAGE 1's CORPUS BACKOFF, fitted BEFORE the registration and BEFORE any bank state is built (design §6.2): the team-level
corpus sampler (c) mixes a consistent corpus team with the joint-Smogon sampler (b) by λ = n / (n + α), n = the corpus
teams consistent with what play revealed. Stage 0 found α = 1 over-confident (its §6.1 point 4). α here is the
maximiser of the never-seen species set's mean log-likelihood under a LEAVE-ONE-TEAM-OUT fit on the LADDER corpus
(``utils.ladder_corpus``, ``full``) with simulated reveals: a team drawn uniformly, k uniform in 1..5 of its mons seen
(a uniform k-subset), each seen mon's moves revealed by a uniform random subset (each move independently with
probability 1/2); the held-out team's own index is removed from the corpus (an identical copy elsewhere stays: ladder
teams recur). No bank state, no pool team and no outcome enters the fit.

    P_c(U*) = λ · f_U + (1 − λ) · P_b(U*)      f_U = the share of the n consistent teams whose never-seen species set
                                              is U*; P_b = (b)'s exact set probability (stage 0's ``p_set_poe``)

Deterministic: the samples are seeded (``SALT``); α is the argmax over a fixed log grid (no optimiser). The FORUM
corpus's own leave-one-team-out α is computed beside it, DESCRIPTIVE only (the registered rule uses the ladder's).

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 python -I fit_alpha.py      -> alpha_fit.json (this folder)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_common as E  # noqa: E402

if str(E.REPO / "src") not in sys.path:
    sys.path.insert(0, str(E.REPO / "src"))
sys.path.insert(1, str(E.STAGE0))
import screen_samplers as S0  # noqa: E402  (stage 0's Smogon prior, (b)'s exact set probability, the corpus reader)

SALT = "ws_stage1_alpha_v1:"
N_FIT = 4000
GRID = np.round(10.0 ** np.arange(-3.0, 4.0001, 0.05), 6)         # 141 points, 0.001 .. 10,000


def simulate(C: "S0.Corpus", S: "S0.Smogon", n: int, salt: str) -> dict:
    """``n`` leave-one-team-out samples on corpus ``C``: per sample (n consistent, f_U, P_b(U*)), plus skip counts."""
    out = {"n": [], "f": [], "pb": [], "k": [], "skip_out_of_pool": 0}
    for i in range(n):
        rng = np.random.default_rng(int(hashlib.sha256(f"{salt}{i}".encode()).hexdigest()[:16], 16))
        t = int(rng.integers(len(C.teams)))
        team = C.teams[t]
        k = int(rng.integers(1, 6))
        seen_ix = sorted(int(x) for x in rng.choice(6, size=k, replace=False))
        revealed = {}
        for j in seen_ix:
            sp, mv = team[j]
            revealed[sp] = {m for m in sorted(mv) if rng.random() < 0.5}
        unseen = frozenset(sp for j, (sp, _) in enumerate(team) if j not in seen_ix)
        if any(s not in S.ix for s in unseen):
            out["skip_out_of_pool"] += 1
            continue
        ids = [c for c in C.consistent(revealed) if c != t]
        nn = len(ids)
        hit = sum(1 for c in ids if frozenset(C.moves[c]) - set(revealed) == unseen)
        sx = [S.ix[s] for s in revealed if s in S.ix]
        pb = S0.p_set_poe(S, sx, [S.ix[s] for s in sorted(unseen)])
        out["n"].append(nn)
        out["f"].append(hit / nn if nn else 0.0)
        out["pb"].append(pb)
        out["k"].append(k)
    return out


def curve(sim: dict) -> np.ndarray:
    n, f, pb = (np.asarray(sim[x], dtype=np.float64) for x in ("n", "f", "pb"))
    ll = []
    for a in GRID:
        lam = n / (n + a)
        p = lam * f + (1.0 - lam) * pb
        ll.append(float(np.mean(np.log(np.maximum(p, 1e-300)))))
    return np.asarray(ll)


def main() -> None:
    t0 = time.time()
    S = S0.Smogon()
    corpora = S0.load_corpora()
    res: dict = {"schema": "ws_stage1_alpha_fit_v1", "salt": SALT, "n_fit": N_FIT,
                 "grid": [float(GRID[0]), float(GRID[-1]), len(GRID)], "smogon_alpha_b": S0.ALPHA}
    for name in ("ladder", "forum"):
        C = corpora[name]
        sim = simulate(C, S, N_FIT, f"{SALT}{name}:")
        ll = curve(sim)
        i = int(np.argmax(ll))
        n = np.asarray(sim["n"])
        res[name] = {
            "corpus_teams": len(C.teams),
            "samples_used": len(sim["n"]), "skip_out_of_pool": sim["skip_out_of_pool"],
            "alpha_star": float(GRID[i]), "mean_logp_at_star": round(float(ll[i]), 5),
            "mean_logp_at_alpha_1": round(float(ll[int(np.argmin(np.abs(GRID - 1.0)))]), 5),
            "mean_logp_b_only": round(float(np.mean(np.log(np.maximum(sim["pb"], 1e-300)))), 5),
            "share_n_zero": round(float(np.mean(n == 0)), 4), "median_n": float(np.median(n)),
            "share_covers_U": round(float(np.mean(np.asarray(sim["f"]) > 0)), 4),
            "curve": {f"{a:g}": round(float(v), 5) for a, v in zip(GRID[::10], ll[::10])},
        }
    res["ALPHA"] = res["ladder"]["alpha_star"]
    res["registered_rule"] = ("λ = n / (n + ALPHA), ALPHA = the LADDER fit's argmax; used by C_ladder AND "
                              "C_forum_strict. The forum fit is descriptive.")
    res["sha256_of_this_script"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    res["wall_s"] = round(time.time() - t0, 1)
    (HERE / "alpha_fit.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("ALPHA",)}, indent=1),
          {n: {k: res[n][k] for k in ("alpha_star", "mean_logp_at_star", "mean_logp_at_alpha_1", "mean_logp_b_only",
                                      "share_n_zero", "share_covers_U")} for n in ("ladder", "forum")})
    if not math.isfinite(res["ladder"]["mean_logp_at_star"]):
        raise SystemExit("the fit's log-likelihood is not finite")


if __name__ == "__main__":
    main()
