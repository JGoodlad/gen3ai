"""Prober: an interactive Textual inspector for saved forensic battle traces.

Browses the ``eval_traces`` tree a training run writes
(``<run_dir>/eval_traces/step_<N>/<opponent>/{win,loss}_<NNN>_{summary.json,states.npz}``)
and analyzes each decision point — faithfulness, type matchups, an intervention
sweep, and gradient saliency — the same analysis the ``probe_replay.py`` CLI
prints for a single invocation, but navigable.

Run:
    export PYTHONPATH=$PYTHONPATH:src
    python -m main.prober <run_dir | eval_traces_dir | summary.json> [--ckpt PATH] [--inv N]
"""

from __future__ import annotations

import argparse

from main.prober.session import ProbeSession

__all__ = ["main", "ProbeSession"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m main.prober",
        description="Interactive forensic-replay inspector for saved battle traces.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="A run dir, an eval_traces dir, or a single *_summary.json.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint .zip override (else auto-resolved from the run dir).",
    )
    parser.add_argument(
        "--inv",
        type=int,
        default=None,
        help="Preselect this invocation index of the first/only battle.",
    )
    args = parser.parse_args()

    # Imported lazily so `--help` doesn't pay the textual/torch import cost.
    from main.prober.app import ProberApp

    ProberApp(root=args.path).run()


if __name__ == "__main__":
    main()
