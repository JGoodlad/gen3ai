#!/usr/bin/env python3
"""Render the README's tables and every PRE-REGISTERED verdict from ``joint_fit.json``.

    python3 report.py --fit <dir>/joint_fit.json [--h2h <dir>/summary.json]

Every P-row from ``PREDICTION.md`` is evaluated HERE, in code, against the registered threshold —
so a prediction cannot quietly become a description of whatever happened. A row prints its
observation, its registered bar and one of PASS / FALSIFIED / REFUTED.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as cells_mod  # noqa: E402

Z95 = 1.959963985


def wilson(k: int, n: int):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + Z95 * Z95 / n
    c = (p + Z95 * Z95 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def elo_gap(p: float) -> float:
    """The Elo difference a win rate implies, for reading a cell beside a fitted rating."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1 - p))


def verdict(ok: bool, refuted: bool = False) -> str:
    return "**REFUTED**" if refuted else ("PASS" if ok else "**FALSIFIED**")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h2h", default=None, help="the anchor-vs-anchor cell's summary.json")
    args = ap.parse_args()
    F = json.loads(Path(args.fit).read_text())
    noext, ext = F["arms"]["JOINT-NOEXT"], F["arms"]["JOINT-EXT"]
    free = F["arms"]["JOINT-EXT-FREEBOT"]
    freerr = F["arms"].get("JOINT-EXT-FREEBOT-RR")
    played = F["played_snapshots"]
    anchors = [n for n in ext["ratings"] if n.startswith("ext:")]
    out = []
    P = out.append

    # ── the cells ────────────────────────────────────────────────────────────────────────────
    P("### Every cell\n")
    P("| our side | anchor | n | W/L/T | win rate (ours) | Wilson 95% | implied Elo gap | mean turns | caps | in fit |")
    P("|---|---|---:|---|---:|---|---:|---:|---:|---|")
    for c in F["cells"]:
        if c["n"] == 0:
            P(f"| `{c['our_node']}` | {c['their_node'].split(':')[-1]} | 0 | — | — | — | — | — | — | "
              f"**NO — {c['exclude_reason']}** |")
            continue
        p, lo, hi = wilson(c["wins"], c["n"])
        P(f"| `{c['our_node']}` | {c['their_node'].split(':')[-1]} | {c['n']} | "
          f"{c['wins']}/{c['losses']}/{c['ties']} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | "
          f"{elo_gap(p):+.0f} | {c['mean_turns']:.1f} | {c['caps']} | "
          + ("yes |" if c["included"] else f"**NO — {c['exclude_reason']}** |"))
    inc = [c for c in F["cells"] if c["included"]]
    P(f"\n{len(inc)}/{len(F['cells'])} cells entered the fit; "
      f"{sum(c['n'] for c in inc)} games, {sum(c['ties'] for c in inc)} ties, "
      f"{sum(c['caps'] for c in inc)} at the 250-turn cap.\n")

    # ── P1 / P2 ──────────────────────────────────────────────────────────────────────────────
    P("### P1 — anchor placement\n")
    P("| anchor | registered | fitted (JOINT-EXT) | se | verdict |")
    P("|---|---|---:|---:|---|")
    reg = {"ext:metamon:SmallRL": 1970.0, "ext:metamon:SyntheticRLV2": 1984.0}
    p1 = True
    for a in sorted(anchors):
        r, se = ext["ratings"][a], ext["se"][a]
        ok = abs(r - reg[a]) <= 100 and se <= 25
        p1 &= ok
        P(f"| `{a}` | {reg[a]:.0f} ± 100, se ≤ 25 | **{r:.1f}** | {se:.1f} | {verdict(ok)} |")
    P(f"\n**P1: {verdict(p1)}**\n")

    snap_cells = [c for c in F["cells"] if c["kind"] == "snapshot" and c["included"]]
    bot_cells = [c for c in F["cells"] if c["kind"] == "bot" and c["included"]
                 and c["our_node"] != "bot:random"]
    med_snap = sorted(abs(c["win_rate"] - 0.5) for c in snap_cells)
    med_snap = med_snap[len(med_snap) // 2] if med_snap else float("nan")
    # The bot SATURATION comparator is our own frontier-vs-bot edge, from the run's eval rows.
    P("### P2 — are the anchors saturated too?\n")
    P(f"median |win rate − 0.5| over the {len(snap_cells)} snapshot cells: **{med_snap:.3f}** "
      f"(registered: < 0.15 expected, FALSIFIED at ≥ 0.25)")
    P(f"\nfor comparison, the same statistic over the {len(bot_cells)} anchor-vs-bot cells: "
      f"**{sorted(abs(c['win_rate'] - 0.5) for c in bot_cells)[len(bot_cells) // 2]:.3f}**, and "
      "our own frontier nodes beat the eight bots at 0.88–0.92 (|wr − 0.5| = 0.38–0.42).")
    P(f"\n**P2: {verdict(med_snap < 0.25)}**"
      + ("" if med_snap < 0.15 else "  (above the 0.15 expectation, inside the 0.25 bar)") + "\n")

    # ── P3 / P4 ──────────────────────────────────────────────────────────────────────────────
    all_snap = [n for n in ext["ratings"] if n.startswith("snap:")]
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    se_before, se_after = mean(noext["se"][n] for n in played), mean(ext["se"][n] for n in played)
    se_all_b, se_all_a = mean(noext["se"][n] for n in all_snap), mean(ext["se"][n] for n in all_snap)
    drop = (se_before - se_after) / se_before if se_before else float("nan")
    P("### P3 — frontier precision\n")
    P("| node set | mean se, JOINT-NOEXT | mean se, JOINT-EXT | change |")
    P("|---|---:|---:|---:|")
    P(f"| the {len(played)} PLAYED nodes | {se_before:.2f} | {se_after:.2f} | {-drop * 100:+.1f}% |")
    P(f"| all {len(all_snap)} snapshot nodes | {se_all_b:.2f} | {se_all_a:.2f} | "
      f"{-(se_all_b - se_all_a) / se_all_b * 100:+.1f}% |")
    P(f"\n**P3: {verdict(drop >= 0.15, refuted=se_after > se_before)}** "
      f"(registered: ≥ 15% reduction on the played nodes; REFUTED if se rises)\n")

    deltas = {n: round(ext["ratings"][n] - noext["ratings"][n], 1) for n in all_snap}
    P("### P4 — how far the scale moved\n")
    P("| node | JOINT-NOEXT | JOINT-EXT | Δ | se before | se after |")
    P("|---|---:|---:|---:|---:|---:|")
    for n in played:
        P(f"| `{n}` | {noext['ratings'][n]:.1f} | {ext['ratings'][n]:.1f} | **{deltas[n]:+.1f}** | "
          f"{noext['se'][n]:.1f} | {ext['se'][n]:.1f} |")
    md = mean(deltas.values())
    md_played = mean(deltas[n] for n in played)
    newest = [f"snap:{r}@{s}" for r, s in cells_mod.SNAPSHOTS][-1:]
    P(f"\nmean Δ over all {len(all_snap)} snapshot nodes: **{md:+.1f} Elo** "
      f"(over the {len(played)} played nodes: {md_played:+.1f}; "
      f"min {min(deltas.values()):+.1f}, max {max(deltas.values()):+.1f})")
    P(f"\n**P4: {verdict(md < 0 and abs(md) < 15)}** "
      "(registered: mean Δ NEGATIVE and |mean Δ| < 15 Elo)\n")

    # ── P5 ───────────────────────────────────────────────────────────────────────────────────
    P("### P5 — newest-node inflation\n")
    P("| run | mean infl, JOINT-NOEXT | mean infl, JOINT-EXT | change | max before | max after |")
    P("|---|---:|---:|---:|---:|---:|")
    p5 = True
    for run, blob in F["inflation"].items():
        b, a = blob["JOINT-NOEXT"]["mean"], blob["JOINT-EXT"]["mean"]
        ch = (b - a) / abs(b) if b else float("nan")
        p5 &= ch >= 0.25
        P(f"| `{run}` | {b:+.1f} | {a:+.1f} | {-ch * 100:+.1f}% | "
          f"{blob['JOINT-NOEXT']['max']:+.1f} | {blob['JOINT-EXT']['max']:+.1f} |")
    P(f"\n**P5: {verdict(p5)}** (registered: ≥ 25% reduction in mean inflation)\n")

    # ── P6 ───────────────────────────────────────────────────────────────────────────────────
    P("### P6 — fit quality\n")
    P("| edge family | JOINT-NOEXT mean\\|err\\| | max | JOINT-EXT mean\\|err\\| | max |")
    P("|---|---:|---:|---:|---:|")
    for fam, key in (("bot (anchor edges to our snapshots)", "residuals_bot_edges"),
                     ("dense (frozen snapshot pairs)", "residuals_dense_edges"),
                     ("external (this campaign)", "residuals_external_edges"),
                     ("ALL", "residuals_all")):
        b, a = noext[key], ext[key]
        P(f"| {fam} | {b['mean_abs_err']} | {b['max_abs_err']} | "
          f"{a['mean_abs_err']} | {a['max_abs_err']} |")
    bot_rise = (ext["residuals_bot_edges"]["mean_abs_err"]
                - noext["residuals_bot_edges"]["mean_abs_err"])
    P(f"\nbot-edge mean|err| moved by **{bot_rise:+.4f}** (registered: must not rise by > 0.01; "
      "`max_abs_err` over ALL edges is ALLOWED to rise — an informative node exposes residuals "
      "the saturated bot edges could not)")
    P(f"\n**P6: {verdict(bot_rise <= 0.01)}**\n")

    # ── P7 ───────────────────────────────────────────────────────────────────────────────────
    P("### P7 — would the PINNED bots move if re-fit jointly?\n")
    P("🚨 **The arm P7 was REGISTERED on is mis-specified, and its own result is what shows it.** "
      "`JOINT-EXT-FREEBOT` frees the eight bots but leaves their OWN round robin — the 36 pairs "
      "and 86,000 games that ARE the pins — out of the edge set. The bots are then held only by "
      "their edges to our snapshots and by `random`'s pin, and `random` loses ~100% of everything, "
      "so its edges are near-perfect scores the Gaussian prior resolves rather than the data. The "
      "cluster slides toward the prior and every bot reads low, uniformly. `JOINT-EXT-FREEBOT-RR` "
      "adds the round robin back and is the arm that answers the question.\n")
    P("| bot | pinned (2026-06-06) | FREEBOT (registered arm) | Δ | **FREEBOT-RR** (correct arm) | **Δ** | se |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    worst = worst_rr = 0.0
    for name, pin in sorted(F["bot_pins"].items(), key=lambda kv: -kv[1]):
        k = f"bot:{name}"
        if k not in free["ratings"]:
            continue
        d = free["ratings"][k] - pin
        worst = max(worst, abs(d))
        if freerr and k in freerr["ratings"]:
            drr = freerr["ratings"][k] - pin
            worst_rr = max(worst_rr, abs(drr))
            rr_cells = (f"{freerr['ratings'][k]:.1f} | **{drr:+.1f}** | "
                        f"{freerr['se'][k]:.1f}")
        else:
            rr_cells = "— | — | —"
        P(f"| `{name}` | {pin:.1f} | {free['ratings'][k]:.1f} | {d:+.1f} | {rr_cells} |")
    P(f"\nlargest move: **{worst:.1f} Elo** in the registered arm (the artifact), "
      f"**{worst_rr:.1f} Elo** in the arm with the round robin (registered bar: < 40 Elo)")
    P(f"\n**P7 as registered: {verdict(worst < 40)}. "
      f"P7 on the corrected arm: {verdict(worst_rr < 40)}.** Reported only — "
      "`data/gen3_bot_elo_anchors.json` and the calibration store are NOT written by this "
      "campaign.\n")

    # ── P8 ───────────────────────────────────────────────────────────────────────────────────
    P("### P8 — the head-to-head\n")
    if args.h2h and Path(args.h2h).exists():
        h = json.loads(Path(args.h2h).read_text())
        p, lo, hi = wilson(h["wins"], h["n"])
        # `win_rate` is the OUR-SIDE player's; our side of that cell is SyntheticRLV2.
        ordered = (ext["ratings"]["ext:metamon:SyntheticRLV2"]
                   > ext["ratings"]["ext:metamon:SmallRL"])
        ok = 0.55 <= p <= 0.75
        P(f"`SyntheticRLV2` vs `SmallRL`, greedy-vs-greedy, home teams, n={h['n']}: "
          f"**{p:.3f}** [{lo:.3f}, {hi:.3f}] (W{h['wins']}/L{h['losses']}/T{h['ties']}), "
          f"implied gap {elo_gap(p):+.0f} Elo")
        P(f"\nthe fit orders them {'the SAME way' if ordered else '**the OTHER way**'} "
          f"(SyntheticRLV2 {ext['ratings']['ext:metamon:SyntheticRLV2']:.1f} vs "
          f"SmallRL {ext['ratings']['ext:metamon:SmallRL']:.1f})")
        P(f"\n**P8: {verdict(ok and ordered)}** (registered: 0.55-0.75 AND the fit agrees)\n")
    else:
        P("**P8: NOT RUN** — the head-to-head cell produced no summary. "
          "The two anchors remain connected through 19 shared opponents.\n")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
