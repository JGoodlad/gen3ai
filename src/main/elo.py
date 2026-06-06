"""Offline ELO / skill-rating analyzer for a training run.

Fits anchored Bradley-Terry ratings (see ``agents.training.elo``) over a run's accumulated
eval results and emits: a ranked ladder table (bots + snapshots, Elo ± 95% CI), an
``elo_ratings.json`` dump, and an ``elo_curve.png`` of snapshot Elo-vs-training-step (the
progress chart that ``win_rate_vs_pool`` cannot give you, because the promotion gate pins it
near 50% by construction).

This is the canonical re-fit — batch-BT is global, so it gives the consistent picture the
live ``eval/elo`` only approximates cycle-by-cycle. It can run on a FINISHED run, and can
**backfill an already-running run straight from TensorBoard** (``--source tb``) with no
training change.

    python -m main.elo <run_dir> [--out DIR] [--source auto|log|tb|meta] [--no-plot]
                       [--anchors data/gen3_bot_elo_anchors.json]

``<run_dir>`` is a ``models/run_<ts>`` directory. ``--out`` defaults to ``<run_dir>/elo/``;
point it elsewhere (e.g. ``/tmp/elo_<ts>``) to analyze a LIVE run without writing into it.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

from agents.training import elo as elo_mod


def _ci95(se: float) -> float:
    return elo_mod.ci95(se)  # single source of the 95% multiplier (agents.training.elo)


def analyze(run_dir: str, source: str, anchors_path: str):
    anchors = elo_mod.load_bot_anchors(anchors_path)
    rows = elo_mod.load_rows(run_dir, source)
    fit = elo_mod.fit_from_run(run_dir, source, anchors_path)
    return rows, fit, anchors


def _print_table(fit: elo_mod.EloFit, anchored: bool) -> None:
    bots = fit.bot_ratings()           # {name: (elo, se)}
    curve = fit.snapshot_curve()       # [(step, elo, se)]
    # One merged ladder, strongest first, bots and snapshots interleaved by rating.
    rows = []
    for name, (elo, se) in bots.items():
        rows.append(("bot", name, elo, se, fit.games.get(elo_mod.bot_key(name), 0),
                     elo_mod.bot_key(name) in fit.pinned))
    for step, elo, se in curve:
        rows.append(("snap", f"{step / 1e6:.1f}M", elo, se,
                     fit.games.get(elo_mod.snap_key(step), 0), False))
    rows.sort(key=lambda r: -r[2])

    print(f"\n{'rank':>4}  {'kind':<5} {'player':<14} {'elo':>7}  {'95% CI':>8}  "
          f"{'games':>7}  anchor")
    print("  " + "─" * 60)
    for i, (kind, label, elo, se, games, pinned) in enumerate(rows, 1):
        anchor = "📌" if pinned else ""
        print(f"{i:>4}  {kind:<5} {label:<14} {elo:7.0f}  ±{_ci95(se):6.0f}  "
              f"{games:>7,}  {anchor}")

    latest = fit.latest_snapshot()
    if latest:
        step, elo, se = latest
        best = max(curve, key=lambda c: c[1])
        print(f"\nlatest snapshot: {step / 1e6:.1f}M → ELO {elo:.0f} ± {_ci95(se):.0f}"
              f"   |   best: {best[0] / 1e6:.1f}M → {best[1]:.0f}")
    anchor_note = ("bot round-robin (pinned)" if anchored else
                   "random=base (fallback — run bot_elo_calibration.py for a "
                   "cross-run-comparable scale)")
    print(f"anchor: {anchor_note}")


def _write_json(fit: elo_mod.EloFit, out_dir: str, source: str, anchored: bool,
                n_cycles: int) -> str:
    payload = {
        "fit": {
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "base": fit.base,
            "source": source,
            "anchored": anchored,
            "n_cycles": n_cycles,
        },
        "snapshots": [
            {"step": s, "elo": round(e, 1), "se": round(se, 1), "ci95": round(_ci95(se), 1),
             "games": fit.games.get(elo_mod.snap_key(s), 0)}
            for s, e, se in fit.snapshot_curve()
        ],
        "bots": [
            {"name": n, "elo": round(e, 1), "se": round(se, 1),
             "games": fit.games.get(elo_mod.bot_key(n), 0),
             "pinned": elo_mod.bot_key(n) in fit.pinned}
            for n, (e, se) in sorted(fit.bot_ratings().items(), key=lambda kv: -kv[1][0])
        ],
    }
    path = os.path.join(out_dir, "elo_ratings.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _write_curve(fit: elo_mod.EloFit, out_dir: str) -> str | None:
    """Best-effort snapshot Elo-vs-step PNG with a 95% CI band + bot anchor lines."""
    curve = fit.snapshot_curve()
    if not curve:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    steps = [s / 1e6 for s, _e, _se in curve]
    elos = [e for _s, e, _se in curve]
    cis = [_ci95(se) for _s, _e, se in curve]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, elos, "-o", color="#1f77b4", label="snapshot ELO", zorder=3)
    ax.fill_between(steps,
                    [e - c for e, c in zip(elos, cis)],
                    [e + c for e, c in zip(elos, cis)],
                    color="#1f77b4", alpha=0.15, zorder=1)
    # Fixed bot anchors as dashed horizontal reference lines.
    for name, (elo, _se) in sorted(fit.bot_ratings().items(), key=lambda kv: kv[1][0]):
        ax.axhline(elo, ls="--", lw=0.8, color="0.6", zorder=0)
        ax.text(steps[-1], elo, f" {name}", va="center", fontsize=7, color="0.4")
    ax.set_xlabel("training step (millions)")
    ax.set_ylabel("ELO (anchored Bradley-Terry)")
    ax.set_title("Model skill rating over training")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "elo_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline ELO analyzer for a training run")
    ap.add_argument("run_dir", help="models/run_<ts> directory")
    ap.add_argument("--out", default=None, help="output dir (default <run_dir>/elo/)")
    ap.add_argument("--source", default="auto", choices=["auto", "log", "tb", "meta"],
                    help="results source (default auto: log → tb → meta)")
    ap.add_argument("--anchors", default=elo_mod.BOT_ANCHORS_PATH,
                    help="bot anchor json (default data/gen3_bot_elo_anchors.json)")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"error: {args.run_dir} is not a directory", file=sys.stderr)
        return 2

    rows, fit, anchors = analyze(args.run_dir, args.source, args.anchors)
    if not rows:
        print(f"No eval results found in {args.run_dir} (source={args.source}).\n"
              "Has the run produced an eval cycle yet? For a run that predates "
              "eval_results.jsonl try --source tb (TensorBoard backfill).", file=sys.stderr)
        return 1

    anchored = bool(anchors)
    out_dir = args.out or os.path.join(args.run_dir, "elo")
    os.makedirs(out_dir, exist_ok=True)

    _print_table(fit, anchored)
    json_path = _write_json(fit, out_dir, args.source, anchored, len(rows))
    print(f"\nwrote {json_path}")
    if not args.no_plot:
        png = _write_curve(fit, out_dir)
        print(f"wrote {png}" if png else
              "curve skipped (matplotlib unavailable or no snapshots yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
