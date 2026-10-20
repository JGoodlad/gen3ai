"""The battery items' TRAINING-TIME (on-distribution) series from X26's TensorBoard, plus the self-play POOL's
composition, at the read points. Descriptive; reads the archive written by ``tb_extract.py``.

    python3 tb_series.py <tb_scalars.json.gz> <out.json> <out.md>

Per tag and read point R: the mean over the trailing window (R - 10M, R] of the per-update values, its SE from the
ten 1M-block means (SE = sd(block means) / sqrt(n blocks): updates are autocorrelated, the block means much less so),
and the 180M -> 380M SLOPE per 100M (OLS over the 1M-block means in (170M, 380.2M]) with a moving-block bootstrap
95 % interval (blocks of 10, 2,000 resamples, seed 20261020). Early context points (30M, 90M, 160M) use the same
trailing window. Nothing here is a registered test; the intervals are descriptive.

The POOL (B-pool, an addition to the battery, see README): X26's self-play pool is the last 20 PROMOTED snapshots
(FIFO; promotion = an eval cycle's ``win_rate_vs_pool`` >= 0.55, ``--promote-threshold``; uniform sampling,
``--pfsp-scale 0``). ``train/selfplay_promoted_steps`` logs every promotion; the pool at step t is the 20 latest
promotions <= t (checked against the run's final ``snapshots/`` directory).
"""
from __future__ import annotations

import gzip
import json
import sys

import numpy as np

READ = [180, 230, 280, 330, 380]
EARLY = [30, 90, 160]
WIN = 10  # M steps, trailing

# (battery item, tag, sign, human description). sign -1 negates (entropy_loss is minus the entropy).
TAGS = [
    ("B1", "train/entropy_loss", -1, "policy entropy on its own rollouts (nats)"),
    ("B5", "ridealong/adv_corr_logit", 1, "A head vs policy logits, correlation"),
    ("B5", "ridealong/adv_explained_var", 1, "A head explained variance of the advantage"),
    ("B5", "ridealong/adv_std_starved", 1, "A members' spread on STARVED actions"),
    ("B5", "ridealong/adv_std_fed", 1, "A members' spread on FED actions"),
    ("B8", "ridealong/rnd_err_mean", 1, "base RND novelty error on rollout states (mean)"),
    ("B8", "ridealong/rnd_z_pool", 1, "base RND z, rows vs the self-play pool"),
    ("B8", "ridealong/rnd_z_bot", 1, "base RND z, rows vs bots"),
    ("B8", "ridealong/rndv_fast_err_mean", 1, "RND fast variant (10x rate: RECENTLY seen) error"),
    ("B8", "ridealong/rndv_decay_err_mean", 1, "RND decay variant error"),
    ("B8", "ridealong/rndv_small_err_mean", 1, "RND small variant error"),
    ("B8", "ridealong/rndv_feat_err_mean", 1, "RND on trunk features error"),
    ("B8", "ridealong/rnd_err_rel_spread", 1, "base RND error relative spread (IQR / median)"),
    ("B9", "opp_intent/flat_loss", 1, "opponent-intent (alpha) loss, all rows"),
    ("B9", "opp_intent/flat_info_gain_nats_pool", 1, "alpha info gain over the prior vs the POOL (nats)"),
    ("B9", "opp_intent/flat_info_gain_nats_bot", 1, "alpha info gain over the prior vs BOTS (nats)"),
    ("B9", "opp_intent/flat_acc_pool", 1, "alpha top-1 accuracy vs the pool"),
    ("B9", "opp_intent/other_label_rate_pool", 1, "OTHER (unmodelled) label rate vs the pool"),
    ("B9", "opp_intent/flat_switch_recall_pool", 1, "alpha switch recall vs the pool"),
    ("B9", "ridealong/opp_loss", 1, "B head (opponent-action contrast) loss"),
    ("B10", "rank/trunk_effrank", 1, "trunk feature effective rank"),
    ("B10", "rank/policy_effrank", 1, "policy feature effective rank"),
    ("B10", "rank/vf_feat_effrank", 1, "value feature effective rank"),
    ("B10", "rank/value_cls_effrank", 1, "value CLS effective rank"),
    ("B10", "train/grad_norm", 1, "gradient norm"),
    ("B10", "grad/policy_value_cosine", 1, "policy-value gradient cosine on the shared trunk"),
    ("upd", "train/learning_rate", 1, "learning rate (adaptive KL controller)"),
    ("upd", "train/approx_kl", 1, "approx KL per update"),
    ("upd", "train/clip_fraction", 1, "PPO clip fraction"),
    ("upd", "train/dose_rate", 1, "dose rate (lr x updates per env step)"),
    ("val", "train/explained_variance", 1, "critic explained variance"),
    ("val", "win_prob/critic_resolution", 1, "critic Brier resolution"),
    ("val", "win_prob/critic_brier", 1, "critic Brier"),
    ("val", "ridealong/ens_brier_v", 1, "V Brier on rollout rows (ensemble reader)"),
    ("opp", "signal/outcome_win_rate_pool", 1, "training win rate vs the self-play pool"),
    ("opp", "signal/outcome_win_rate_bots", 1, "training win rate vs bots"),
    ("opp", "signal/draw_rate", 1, "training draw rate"),
    ("opp", "rollout/ep_len_mean", 1, "episode length (decisions)"),
    ("opp", "eval/win_rate_vs_pool", 1, "EVAL greedy win rate vs the 5 sentinels"),
    ("opp", "eval/win_rate_vs_bots", 1, "EVAL greedy win rate vs bots"),
]


def blocks(s: np.ndarray, lo_m: float, hi_m: float) -> np.ndarray:
    """1M-block means of the (step, value) rows with lo < step <= hi (steps in M)."""
    st, v = s[:, 0] / 1e6, s[:, 1]
    keep = (st > lo_m) & (st <= hi_m)
    st, v = st[keep], v[keep]
    if st.size == 0:
        return np.zeros((0, 2))
    b = np.floor(st - 1e-9).astype(int)
    out = [(k + 0.5, float(v[b == k].mean())) for k in np.unique(b)]
    return np.array(out)


def window(s: np.ndarray, r: int) -> dict:
    hi = r + 0.2 if r == 380 else r  # the run's last update is 380.13M
    bm = blocks(s, r - WIN, hi)
    if len(bm) < 2:
        return {"mean": None, "se": None, "n_blocks": int(len(bm))}
    return {"mean": float(bm[:, 1].mean()), "se": float(bm[:, 1].std(ddof=1) / np.sqrt(len(bm))),
            "n_blocks": int(len(bm))}


def slope(s: np.ndarray, rng: np.random.Generator) -> dict:
    bm = blocks(s, 170, 380.2)
    x, y = bm[:, 0] / 100.0, bm[:, 1]
    b = float(np.polyfit(x, y, 1)[0])
    n, L = len(x), 10
    boots = []
    for _ in range(2000):
        idx = np.concatenate([np.arange(st, st + L) for st in rng.integers(0, n - L + 1, size=n // L + 1)])[:n]
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"per_100M": b, "ci95": [float(lo), float(hi)], "n_blocks": n}


def pool_series(promoted: list[float]) -> dict:
    prom = sorted(promoted)
    out = {}
    for r in EARLY + READ:
        mem = [p for p in prom if p <= r * 1e6][-20:]
        ages = [(r * 1e6 - p) / 1e6 for p in mem]
        in_50 = [p for p in prom if (r - 50) * 1e6 < p <= r * 1e6]
        out[str(r)] = {"pool_size": len(mem), "oldest_M": round(min(mem) / 1e6, 1) if mem else None,
                       "newest_M": round(max(mem) / 1e6, 1) if mem else None,
                       "age_median_M": round(float(np.median(ages)), 1) if ages else None,
                       "age_mean_M": round(float(np.mean(ages)), 1) if ages else None,
                       "promotions_in_last_50M": len(in_50)}
    return out


def main(src: str, out_json: str, out_md: str) -> int:
    tags = json.load(gzip.open(src, "rt"))["tags"]
    rng = np.random.default_rng(20261020)
    res: dict = {"schema": "x26_collapse_battery_tb_v1", "window_M": WIN, "read_points_M": READ,
                 "early_points_M": EARLY, "tags": {}}
    lines = ["| item | tag (what) | 30M | 90M | 160M | 180M | 230M | 280M | 330M | 380M | slope 180→380 per 100M [95 %] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for item, tag, sign, what in TAGS:
        if tag not in tags:
            res["tags"][tag] = {"missing": True}
            continue
        s = np.array([[r[0], sign * r[1]] for r in tags[tag]], dtype=float)
        s = s[np.isfinite(s[:, 1])]
        w = {str(r): window(s, r) for r in EARLY + READ}
        sl = slope(s, rng)
        res["tags"][tag] = {"item": item, "what": what, "sign": sign, "windows": w, "slope": sl}

        def f(d):
            if d["mean"] is None:
                return "—"
            m, se = d["mean"], d["se"]
            if 0 < abs(m) < 0.01:
                return f"{m:.3g} ±{se:.1g}"
            dig = 4 if abs(m) < 1 else (3 if abs(m) < 10 else 1)
            return f"{m:.{dig}f} ±{se:.{dig}f}"

        def g(x):
            return f"{x:+.3g}"
        lines.append(f"| {item} | `{tag}` ({what}) | " + " | ".join(f(w[str(r)]) for r in EARLY + READ)
                     + f" | {g(sl['per_100M'])} [{g(sl['ci95'][0])}, {g(sl['ci95'][1])}] |")
    prom = [r[1] for r in tags["train/selfplay_promoted_steps"]]
    res["pool"] = pool_series(prom)
    res["pool"]["promotions"] = [round(p / 1e6, 1) for p in sorted(prom)]
    lines += ["", "| pool at | 30M | 90M | 160M | 180M | 230M | 280M | 330M | 380M |", "|---|---|---|---|---|---|---|---|---|"]
    for k, lab in (("oldest_M", "oldest member (M)"), ("newest_M", "newest member (M)"),
                   ("age_median_M", "median member age (M steps)"),
                   ("promotions_in_last_50M", "promotions in the last 50M")):
        lines.append(f"| {lab} | " + " | ".join(str(res["pool"][str(r)][k]) for r in EARLY + READ) + " |")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
