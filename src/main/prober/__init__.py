"""Prober: a forensic inspector for saved battle traces.

Browses the ``eval_traces`` tree a training run writes
(``<run_dir>/eval_traces/step_<N>/<opponent>/{win,loss}_<NNN>_{summary.json,states.npz}``)
and analyzes each decision point — faithfulness, beliefs, threat tables, an
intervention sweep, and gradient saliency.

**The interface is the browser** (``main.prober.web``). The Textual TUI that used to live in
``main.prober.app`` is GONE: one analysis engine deserves one renderer, and two meant every new
signal had to be drawn twice for a single reader. ``python -m main.prober <run>`` therefore starts
the web app rather than a terminal UI — the muscle memory keeps working, the surface changed.

Run:
    export PYTHONPATH=$PYTHONPATH:src
    python -m main.prober <run_dir | models_dir>      # -> http://127.0.0.1:6008
    python -m main.prober.query <cmd> ...             # the JSON CLI, for agents and scripts
"""

from __future__ import annotations

from main.prober.session import ProbeSession

__all__ = ["main", "ProbeSession"]


def main() -> None:
    """Start the browser front end. This used to launch a Textual TUI; the TUI is retired.

    Kept as an alias rather than deleted because `python -m main.prober <run>` is in muscle memory,
    in scripts and throughout the docs — silently 404-ing that would be a worse migration than
    changing what it opens. Every argument is forwarded to `main.prober.web`, so `--port` /
    `--impl` / `--open` work here too; the two TUI-only flags are translated below.
    """
    import sys

    argv = list(sys.argv[1:])
    # `--ckpt` / `--inv` were TUI selection flags with no web equivalent (the web resolves the
    # checkpoint per battle, and a decision is addressed by URL). Say so rather than failing on an
    # unrecognised argument, which would read as a broken command.
    for dead, instead in (("--ckpt", "the web resolves the checkpoint per battle "
                                     "(exact -> nearest -> recent)"),
                          ("--inv", "a decision is addressed by URL: /analyze?battle=<id>&inv=<n>")):
        if dead in argv:
            i = argv.index(dead)
            del argv[i:i + 2]
            print(f"note: {dead} is gone with the TUI — {instead}", file=sys.stderr)

    from main.prober.web.__main__ import main as web_main
    raise SystemExit(web_main(argv))


if __name__ == "__main__":
    main()
