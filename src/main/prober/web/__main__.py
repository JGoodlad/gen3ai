"""Serve the prober web views.

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.prober.web models/run_<timestamp>

Binds loopback on 6008 by default (beside tensorboard's 6006 and the arch viewer's 6007). Put a
tunnel in front of it rather than binding a public interface — same posture as
`arch_viewer_serve.py`, which is what fronts model.g5d.io.

`--openapi <path>` writes the contract instead of serving, which is how the committed
`openapi.json` snapshot is regenerated; `--check-openapi` is the staleness gate.
"""

from __future__ import annotations

import argparse
import json
import os

_DEFAULT_SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openapi.json")


def openapi_document() -> dict:
    """The contract, generated from an app with NO run directory.

    Generating it data-free is the point: the snapshot must describe the API surface, so it cannot
    be allowed to vary with whatever happens to be in someone's `models/`.
    """
    from main.prober.web.app import create_app
    return create_app(None).openapi()


def _canonical(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.prober.web",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("run", nargs="?", default=None, metavar="PATH",
                    help="a models/ directory (the picker enumerates its runs) or a single run "
                         "dir (the picker then offers only that one)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default loopback)")
    ap.add_argument("--port", type=int, default=6008, help="port (default 6008)")
    ap.add_argument("--job-workers", type=int, default=2,
                    help="concurrent falsify/calibration jobs (default 2 — each spawns Node)")
    ap.add_argument("--open", dest="open_access", action="store_true",
                    help="no password: anyone may start the expensive probes. For a laptop, "
                         "NOT for anything reachable off this box.")
    ap.add_argument("--openapi", nargs="?", const=_DEFAULT_SNAPSHOT, default=None,
                    metavar="PATH", help="write the OpenAPI contract and exit "
                                         f"(default {os.path.basename(_DEFAULT_SNAPSHOT)})")
    ap.add_argument("--check-openapi", action="store_true",
                    help="exit 1 if the committed openapi.json is stale (drift gate)")
    a = ap.parse_args(argv)

    if a.check_openapi:
        want = _canonical(openapi_document())
        if not os.path.exists(_DEFAULT_SNAPSHOT):
            print(f"MISSING {_DEFAULT_SNAPSHOT}")
            return 1
        with open(_DEFAULT_SNAPSHOT) as fh:
            cur = fh.read()
        if cur != want:
            print(f"STALE {_DEFAULT_SNAPSHOT} — regenerate with: "
                  "python -m main.prober.web --openapi")
            return 1
        print(f"OK {_DEFAULT_SNAPSHOT} is current")
        return 0

    if a.openapi:
        with open(a.openapi, "w") as fh:
            fh.write(_canonical(openapi_document()))
        print(f"wrote {a.openapi}")
        return 0

    if a.run is None:
        ap.error("a run directory is required to serve (or use --openapi / --check-openapi)")
    # `build_trace_tree` TOLERATES a path with no traces — it returns an empty tree rather than
    # raising. On the CLI that is fine (you see an empty JSON object); on a web page it renders as
    # a confident "0 battles captured", which reads as a finding about the run rather than a typo
    # in the path. So the check lives here, at startup, where it can still be a hard error.
    if not os.path.exists(a.run):
        ap.error(f"no such run path: {a.run}")

    import uvicorn

    from main.prober.web.app import create_app
    from main.prober.web.auth import ENV_FILE_VAR, ENV_VAR, load_password

    password = load_password()
    app = create_app(os.path.abspath(a.run), max_job_workers=a.job_workers,
                     password=password, open_access=a.open_access)
    n_runs = len(app.state.runs.list_runs())

    print(f"prober web → http://{a.host}:{a.port}   ({n_runs} run(s) under {a.run})")
    # Say the access posture out loud at startup. A hosted instance silently running with the
    # probes wide open, or silently unable to run them at all, are both states an operator should
    # learn from the log rather than from a user's confusion.
    if a.open_access:
        print("  probes: OPEN — no password required. Do not expose this beyond loopback.")
    elif password:
        print("  probes: locked behind the shared password (reading stays anonymous)")
    else:
        print(f"  probes: DISABLED — no password set. Set {ENV_VAR} or {ENV_FILE_VAR} "
              f"(reading still works).")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
