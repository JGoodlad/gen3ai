"""A browser front end for the prober — a THIRD sibling over the analysis engine.

`engine.py` is the analysis; `app.py` (Textual) and `query.py` (JSON CLI) are two independent
callers of it. This package is a third: a FastAPI app whose handlers are a thin adapter over
`ProbeSession`, the same facade `query.py` uses. It adds **no analysis of its own** — every number
on every page comes back from a `ProbeSession` method verbatim.

Nothing here imports the TUI, and `ProbeSession` is unmodified.

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.prober.web <run_dir>            # http://127.0.0.1:6008

Read-only views (the ones a terminal renders worst, where the JSON already exists):
run summary · battles · cross-battle `scan` · loss-lever `triage` · the `falsify_scan`
crater bracket · the critic `calibration` reliability curve.
"""

from main.prober.web.app import create_app

__all__ = ["create_app"]
