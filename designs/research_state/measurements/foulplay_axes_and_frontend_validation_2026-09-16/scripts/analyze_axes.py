#!/usr/bin/env python3
"""The registered analysis of the four Foul Play cells.

Reads each cell's games.jsonl + summary.json, and prints:
  * the per-cell table (win rate, Wilson 95%, realized visits, turns, integrity counters);
  * the WIDTH SLOPE — OLS of win rate on ln(realized visits/decision) over the three
    standard-knowledge cells, with a percentile interval from a game-level bootstrap;
  * the WIDTH CONTRAST w100 - w1000 with a Newcombe 95% interval;
  * the KNOWLEDGE CONTRAST k1000 - w1000, and its WIDTH-CORRECTED residual.

Nothing here is fitted after seeing the knowledge cell: the slope uses only the three
standard-knowledge cells, exactly as PREDICTION.md §5 registers.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

CELLS = ["w100", "w300", "w1000", "k1000", "k100"]
ROOT = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/fp_axes/cells")
VISITS_RE = re.compile(r"Iterations\s+\d+:\s+([\d.eE+]+)")


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Newcombe's method 10 interval for p1 - p2 (independent samples)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def load(cell: str) -> dict:
    d = ROOT / cell
    rows = [json.loads(x) for x in (d / "games.jsonl").read_text().splitlines() if x.strip()]
    summ = json.loads((d / "summary.json").read_text())
    visits: list[float] = []
    for log in sorted(d.glob("*/peer_*.log")):
        visits += [float(m) for m in VISITS_RE.findall(log.read_text(errors="replace"))]
    per_half = {}
    for half in ("ours_challenge", "peer_challenge"):
        f = d / half
        vs: list[float] = []
        for log in sorted(f.glob("peer_*.log")):
            vs += [float(m) for m in VISITS_RE.findall(log.read_text(errors="replace"))]
        if vs:
            per_half[half] = statistics.mean(vs)
    meta = (ROOT / f"{cell}.meta").read_text() if (ROOT / f"{cell}.meta").exists() else ""
    return {
        "cell": cell, "rows": rows, "summary": summ, "visits": visits,
        "visits_by_half": per_half, "meta": meta,
    }


def main() -> None:
    data = {c: load(c) for c in CELLS if (ROOT / c / "summary.json").exists()}
    print("## Per-cell\n")
    hdr = ("| cell | status | n | W/L/T | win rate | Wilson 95% | realized visits/dec "
           "(mean) | median | turns mean/med/max | forfeits | asym |")
    print(hdr)
    print("|---|---|---:|---|---:|---|---:|---:|---|---:|---|")
    for c, d in data.items():
        s = d["summary"]
        n, w = s["n"], s["wins"]
        lo, hi = wilson(w, n)
        v = d["visits"]
        print(f"| `{c}` | {s['status']} | {n} | {w}/{s['losses']}/{s.get('ties', 0)} | "
              f"**{w / n:.3f}** | [{lo:.3f}, {hi:.3f}] | "
              f"{statistics.mean(v):,.0f} | {statistics.median(v):,.0f} | "
              f"{s['mean_turns']:.1f}/{s['median_turns']}/{s['max_turns']} | "
              f"{s['hit_forfeit_limit']} | {s['team_source_asymmetry']} |")

    std = [c for c in ("w100", "w300", "w1000") if c in data]
    if len(std) >= 2:
        xs = [math.log(statistics.mean(data[c]["visits"])) for c in std]
        ys = [data[c]["summary"]["wins"] / data[c]["summary"]["n"] for c in std]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        intercept = my - slope * mx
        print(f"\n## Width slope (standard-knowledge cells only: {', '.join(std)})\n")
        for c, x, y in zip(std, xs, ys):
            print(f"  {c}: ln(visits)={x:.4f} (={math.exp(x):,.0f})  win rate={y:.3f}")
        rng = random.Random(20260917)
        boots = []
        for _ in range(4000):
            bys = []
            for c in std:
                rows = data[c]["rows"]
                n = len(rows)
                bys.append(sum(1 for _ in range(n) if rng.choice(rows)["won"]) / n)
            bmy = statistics.mean(bys)
            boots.append(sum((x - mx) * (y - bmy) for x, y in zip(xs, bys)) / den)
        boots.sort()
        blo, bhi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        print(f"\n  **slope = {slope:+.4f} win rate per ln-unit of visits**, "
              f"game-bootstrap 95% [{blo:+.4f}, {bhi:+.4f}]")
        print(f"  (= {slope * math.log(2):+.4f} per DOUBLING of realized width); "
              f"intercept {intercept:+.4f}")

    if "w100" in data and "w1000" in data:
        a, b = data["w100"]["summary"], data["w1000"]["summary"]
        d_, lo, hi = newcombe(a["wins"], a["n"], b["wins"], b["n"])
        verdict = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
        print(f"\n## Width contrast  w100 - w1000 = **{d_:+.3f}**  Newcombe 95% "
              f"[{lo:+.3f}, {hi:+.3f}] — **{verdict}**")

    if "k1000" in data and "w1000" in data:
        a, b = data["k1000"]["summary"], data["w1000"]["summary"]
        d_, lo, hi = newcombe(a["wins"], a["n"], b["wins"], b["n"])
        verdict = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
        print(f"\n## Knowledge contrast  k1000 - w1000 = **{d_:+.3f}**  Newcombe 95% "
              f"[{lo:+.3f}, {hi:+.3f}] — **{verdict}** (uncorrected)")
        vk = math.log(statistics.mean(data["k1000"]["visits"]))
        vw = math.log(statistics.mean(data["w1000"]["visits"]))
        corr = slope * (vk - vw)
        print(f"\n  realized ln-width: k1000 {vk:.4f} ({math.exp(vk):,.0f}), "
              f"w1000 {vw:.4f} ({math.exp(vw):,.0f}); "
              f"the width slope predicts {corr:+.4f} of that difference from width alone")
        rd, rlo, rhi = d_ - corr, lo - corr, hi - corr
        rverdict = "DETECTED" if (rlo > 0 or rhi < 0) else "NOT DETECTED"
        print(f"\n  **width-corrected knowledge residual = {rd:+.3f}  95% "
              f"[{rlo:+.3f}, {rhi:+.3f}] — {rverdict}**")

    if "k100" in data and "w100" in data:
        a, b = data["k100"]["summary"], data["w100"]["summary"]
        d_, lo, hi = newcombe(a["wins"], a["n"], b["wins"], b["n"])
        v = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
        print(f"\n## EXPLORATORY knowledge contrast at 100 ms  k100 - w100 = **{d_:+.3f}**  "
              f"Newcombe 95% [{lo:+.3f}, {hi:+.3f}] — **{v}** (NOT pre-registered)")
        kw = sum(data[c]["summary"]["wins"] for c in ("k100", "k1000"))
        kn = sum(data[c]["summary"]["n"] for c in ("k100", "k1000"))
        ww = sum(data[c]["summary"]["wins"] for c in ("w100", "w1000"))
        wn = sum(data[c]["summary"]["n"] for c in ("w100", "w1000"))
        d_, lo, hi = newcombe(kw, kn, ww, wn)
        v = "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED"
        print(f"  pooled over the two matched widths: degraded {kw}/{kn} = {kw / kn:.3f} vs "
              f"standard {ww}/{wn} = {ww / wn:.3f}; delta {d_:+.3f} [{lo:+.3f}, {hi:+.3f}] — {v}")

    print("\n## Paired games (same our-team index across cells)\n")
    for pair in (("w100", "w1000"), ("k1000", "w1000"), ("k100", "w100")):
        if not all(c in data for c in pair):
            continue
        A = {(r["half"], r["index"]): r for r in data[pair[0]]["rows"]}
        B = {(r["half"], r["index"]): r for r in data[pair[1]]["rows"]}
        both = [k for k in A if k in B and A[k]["our_team"] == B[k]["our_team"]]
        if not both:
            print(f"  {pair[0]} vs {pair[1]}: 0 game indices share our team — no paired read")
            continue
        wins_a = sum(1 for k in both if A[k]["won"])
        wins_b = sum(1 for k in both if B[k]["won"])
        disc_ab = sum(1 for k in both if A[k]["won"] and not B[k]["won"])
        disc_ba = sum(1 for k in both if B[k]["won"] and not A[k]["won"])
        n = len(both)
        d_ = (wins_a - wins_b) / n
        # McNemar exact-ish: the discordant pairs carry the whole signal
        m = disc_ab + disc_ba
        se = math.sqrt(m) / n if m else float("nan")
        print(f"  {pair[0]} vs {pair[1]}: {n} matched-team game slots, "
              f"{wins_a} vs {wins_b} wins, paired Δ = {d_:+.3f}; "
              f"discordant {disc_ab}/{disc_ba} (m={m}, se≈{se:.3f})")

    print("\n## Provenance\n")
    for c, d in data.items():
        s = d["summary"]
        pr = s.get("provenance", {})
        print(f"  {c}: opponent_commit={s['cell']['opponent_commit']} "
              f"showdown_pin={s['cell']['showdown_pin']} "
              f"model_step={s['cell']['model_step']} "
              f"loader={s['cell']['model_loader']} "
              f"their_argmax={s.get('their_argmax_match_rates')} "
              f"wall_s={pr.get('peer_report', {}).get('wall_s')}")
        print("    " + " | ".join(x.strip() for x in d["meta"].splitlines() if "CELL" in x))
        print(f"    visits by half: " + ", ".join(
            f"{h}={v:,.0f}" for h, v in d["visits_by_half"].items()))


if __name__ == "__main__":
    sys.exit(main())
