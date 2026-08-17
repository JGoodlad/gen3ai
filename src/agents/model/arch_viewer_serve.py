"""Serve the delivery digraph live from a checkout — the headless-workstation path.

Split out of `build_arch_viewer` because generating the artifact and serving it are different
jobs with different failure modes: the generator is a pure function of the tree that must be
byte-deterministic, while this is a long-lived process with a socket, a cache-validation story
and a hot-reload story. Mixing them put an HTTP server in the middle of a module whose other
callers only ever wanted a string.

The unit that runs this is `scripts/workstation/gen3ai-model-viewer.service`; the operational
notes (backoff, the restart caveat, Cloudflare Access) live in
`scripts/workstation/GCP_INFRASTRUCTURE.md`.
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import os
import traceback
from typing import Any, Dict, Optional

from agents.model import build_arch_viewer as _B

_SNAPSHOT = _B._SNAPSHOT
_REPO = _B._REPO


def _snapshot_provenance() -> Optional[Dict[str, Any]]:
    """When was the snapshot last regenerated? Served pages only.

    Not embedded in the committed artifact on purpose: git does not preserve mtimes, so a date
    baked into the HTML would differ on every clone and break the byte-comparison `--check`
    depends on. A live server, though, is sitting on a real checkout and can just look.

    Prefers the snapshot's last COMMIT date over its mtime — mtime tells you when the file was
    written (a `git pull` rewrites it), the commit date tells you when the architecture was
    actually last rebuilt, which is the question being asked.
    """
    import datetime
    import subprocess

    iso, source = None, "mtime"
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", _SNAPSHOT],
                             cwd=_REPO, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            iso, source = out.stdout.strip(), "git"
    except Exception:                              # noqa: BLE001 - git absent / not a checkout
        pass
    if iso is None:
        try:
            iso = datetime.datetime.fromtimestamp(
                os.path.getmtime(_SNAPSHOT)).astimezone().isoformat()
        except OSError:
            return None
    try:
        when = datetime.datetime.fromisoformat(iso)
        days = (datetime.datetime.now(when.tzinfo) - when).days
    except ValueError:
        return None
    return {"iso": iso, "days": days, "source": source}


def _live_module(_state: Dict[str, Any] = {}) -> Any:
    """Hand back THIS module, reloaded whenever its source changes. No restart, ever.

    The graph snapshot, the measurement files and the `_EDGE_*_CELL` block are already re-read per
    request, but the HTML template and `FAMILY_LABEL` are module globals bound at import — so
    without this, editing the generator kept serving the old page until someone restarted the unit.
    Now every input is live and the service only ever needs restarting for a Python upgrade.

    Loaded into a FRESH module object and swapped in only on success. `importlib.reload()` would be
    the obvious call and is the wrong one: it re-executes into the EXISTING namespace, so a file
    caught mid-save — half a function, a syntax error — corrupts the module you are still serving
    from. A fresh object either builds or is discarded, and the old one keeps working either way.
    """
    path = _B.__file__
    mtime = os.path.getmtime(path)
    if _state and _state["mtime"] == mtime:
        return _state["mod"]
    if not _state:                                 # first call: adopt the imported module
        _state.update(mod=_B, mtime=mtime)
        return _state["mod"]
    spec = importlib.util.spec_from_file_location(_B.__name__ + "_live", path)
    # a real .py path always resolves a spec and a loader; None would mean the file vanished
    mod = importlib.util.module_from_spec(spec)    # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]  # a broken edit raises here; _state is untouched
    _state.update(mod=mod, mtime=mtime)            # ...so we only commit to a module that built
    return mod


def serve(port: int, host: str = "127.0.0.1") -> int:
    """Serve the viewer, RE-RENDERED FROM DISK on every request.

    This is the answer to "how does the endpoint stay up to date", and it answers it by removing
    the question. Nothing is uploaded and nothing is cached: each GET rebuilds the payload from the
    checkout this process is running in, so the page cannot be more stale than the working tree.
    A deployed copy — Pages, an rsync, a file dropped on a webserver — is a SECOND artifact that
    rots the moment the first one moves, which is the failure this whole module exists to prevent.

    Cheap enough to do per-request: the default path reads JSON and one source file, never imports
    torch, and renders in ~6 ms (measured). (`--config` does import `delivery_graph`, which is why
    the served build always uses the committed snapshot.)

    The RENDERING path is fully live, including this file — see `_live_module`: the snapshot, the
    measurements, the `_EDGE_*_CELL` block, the HTML template and `FAMILY_LABEL` all reach a
    request with no restart. What does NOT hot-reload is the server itself: `serve` and `Handler`
    are already-bound closures, so a change to the ROUTES or to what the handler injects needs
    `systemctl --user restart gen3ai-model-viewer`. Measured, not assumed — an architecture change
    reached the live endpoint untouched while a new handler-injected field did not.

    ETag + `Cache-Control: no-cache` rather than `no-store`: the browser revalidates on every load
    (so it can never show you a stale architecture) but gets a 304 and zero bytes back when nothing
    has changed. `no-store` would forbid the cache entirely and re-send 300 KB for every refresh,
    which buys nothing — the freshness comes from re-rendering, not from refusing to cache.

    Binds to LOOPBACK by default. The intended front door is the existing Cloudflare Tunnel, which
    connects from the same box; binding to 0.0.0.0 would put the architecture and its audit numbers
    on the LAN with no auth, which is not a decision this flag should make for you.

    Routes:
      /            the viewer
      /graph.json  the raw payload — the better thing to hand an AI than 300 KB of HTML
      /healthz     "ok" plus the node/edge counts, for a probe
    """
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gen3ai-arch-viewer"

        def _send(self, code: int, body: bytes, ctype: str,
                  etag: Optional[str] = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            if etag:
                self.send_header("ETag", etag)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _body(self, body: bytes, ctype: str) -> None:
            """Serve with an ETag, answering 304 when the caller already has these bytes."""
            etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(200, body, ctype, etag)

        def do_GET(self) -> None:                  # noqa: N802 - stdlib naming
            route = self.path.split("?")[0].split("#")[0]
            try:
                mod = _live_module()
                if route in ("/", "/index.html"):
                    payload = mod.build_payload(mod._load_graph(None))
                    payload["snapshot_built"] = _snapshot_provenance()
                    self._body(mod.render(payload).encode(), "text/html; charset=utf-8")
                elif route == "/graph.json":
                    payload = mod.build_payload(mod._load_graph(None))
                    payload["snapshot_built"] = _snapshot_provenance()
                    self._body(json.dumps(payload, indent=1).encode(),
                               "application/json; charset=utf-8")
                elif route == "/healthz":
                    p = mod.build_payload(mod._load_graph(None))
                    self._send(200,
                               f"ok {len(p['nodes'])} nodes {len(p['edges'])} edges\n".encode(),
                               "text/plain; charset=utf-8")
                else:
                    self._send(404, b"not found\n", "text/plain; charset=utf-8")
            except Exception:                      # noqa: BLE001
                # Fail LOUD and in place. A 500 with the traceback is worth far more than a page
                # that quietly keeps showing the last architecture that happened to build — and it
                # is what you want mid-edit, when the file on disk genuinely is broken.
                self._send(500, ("the viewer failed to render from this checkout:\n\n"
                                 + traceback.format_exc()).encode(),
                           "text/plain; charset=utf-8")

        do_HEAD = do_GET                           # noqa: N815 - stdlib naming

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            pass                                   # one line per request is just noise here

    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"serving the delivery digraph on http://{host}:{port}")
    print(f"  rendering live from {_REPO} — no build artifact, no restart on change")
    print("  routes: /  ·  /graph.json  ·  /healthz")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0
