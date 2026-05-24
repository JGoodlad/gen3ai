#!/usr/bin/env python3
"""CLI fallback for generating training-run plots.

Usage:
    # Derive tensorboard dir from run dir automatically
    python src/utils/plot_run.py models/run_20260521_091605/

    # Explicit tensorboard dir (outputs to ./tb_imgs/)
    python src/utils/plot_run.py --tb tensorboard/MPPO_run_20260521_091605_0/

    # Auto-find the most recent run dir
    python src/utils/plot_run.py --latest

    # Override EMA smoothing weight (default 0.75, same as TensorBoard default)
    python src/utils/plot_run.py --latest --smoothing 0.9

Output always goes to <run_dir>/tb_imgs/, one PNG per metric group.
"""

import argparse
import glob
import os
import sys


def _find_latest_run_dir() -> str | None:
    """Return the most recently modified models/run_*/ directory."""
    try:
        from utils.git import get_main_repo_root
        repo_root = get_main_repo_root()
    except Exception:
        repo_root = os.getcwd()

    models_root = os.path.join(repo_root, "models")
    candidates = [
        m for m in glob.glob(os.path.join(models_root, "run_*"))
        if os.path.isdir(m)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TensorBoard metric plots as PNGs."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "run_dir",
        nargs="?",
        help="Path to a model run directory (e.g. models/run_20260521_091605/).",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Auto-find the most recent run directory.",
    )
    parser.add_argument(
        "--tb",
        metavar="TB_DIR",
        help="Explicit TensorBoard directory (overrides auto-detection). "
             "Output goes to ./tb_imgs/ when no run_dir is known.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.75,
        metavar="W",
        help="EMA smoothing weight in [0, 1]. Default: 0.75 (matches TensorBoard).",
    )
    args = parser.parse_args()

    # Resolve run directory.
    run_dir: str | None = None
    if args.latest:
        run_dir = _find_latest_run_dir()
        if not run_dir:
            print("ERROR: No run_* directories found under models/.", file=sys.stderr)
            sys.exit(1)
        print(f"Latest run: {run_dir}")
    elif args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        if not os.path.isdir(run_dir):
            print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

    # Resolve TensorBoard directory.
    tb_dir: str | None = args.tb
    if tb_dir is None and run_dir is not None:
        from utils.plot_tb import find_tb_dir_for_run
        tb_dir = find_tb_dir_for_run(run_dir)
        if tb_dir is None:
            print(
                f"ERROR: Could not find a TensorBoard directory for {run_dir}.\n"
                "Pass --tb <dir> to specify it explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"TensorBoard dir: {tb_dir}")

    if tb_dir is None:
        print(
            "ERROR: Provide a run_dir, --latest, or --tb <dir>.",
            file=sys.stderr,
        )
        parser.print_usage(sys.stderr)
        sys.exit(1)

    tb_dir = os.path.abspath(tb_dir)
    if not os.path.isdir(tb_dir):
        print(f"ERROR: TensorBoard directory not found: {tb_dir}", file=sys.stderr)
        sys.exit(1)

    # Output directory: <run_dir>/tb_imgs/ or ./tb_imgs/ if no run_dir.
    if run_dir is not None:
        out_dir = os.path.join(run_dir, "tb_imgs")
    else:
        out_dir = os.path.join(os.getcwd(), "tb_imgs")

    print(f"Generating plots → {out_dir} ...")
    from utils.plot_tb import generate_plots
    paths = generate_plots(tb_dir, out_dir, smoothing=args.smoothing)

    if paths:
        print(f"Wrote {len(paths)} PNG(s):")
        for p in paths:
            print(f"  {p}")
    else:
        print("No scalar data found — nothing written.")


if __name__ == "__main__":
    main()
