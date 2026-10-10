"""The unlock GATE — the one mechanism that keeps the app's expensive work behind the shared password.

`auth.py` states the rule: reading needs nothing, WORK needs the password, because a public endpoint
that starts work is a free CPU-burn button beside a live training run. "Work" is two things:

  * `kind="job"`   — a background probe (falsify_scan, calibration, lookahead, better-line,
                     replay-counterfactual): minutes of Node re-rolls.
  * `kind="model"` — anything that LOADS A CHECKPOINT or RUNS A MODEL FORWARD (`/analyze`, `/game`'s
                     model panels): seconds of CPU and a ~27 MB model resident per request, and the
                     step is chosen by an anonymous query string.

A route is gated by carrying `job_gate(...)` / `model_gate(...)` in its `dependencies=[...]`. That is
the ONLY form: no handler checks `unlocked()` by hand any more, so "is this route gated" is one
question the route table answers (`route_is_gated`), and the CLASS GUARD
(`web/gate_guard_test.py`) asks it of every route whose handler reaches model-loading code.

A locked request to an `/api/...` or page route is a plain 403; to a `/partials/...` HTMX target it is
a 200 fragment carrying the reason and the way forward (HTMX swallows non-2xx into a generic error,
which would read as a broken page, not as "unlock to view").
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request

from main.prober.web.auth import COOKIE

#: set on the dependency callable; `route_is_gated` reads it off the route table.
GATE_ATTR = "gen3ai_unlock_gate"

_DETAIL = {
    "job": "this probe spends minutes of CPU beside a live training run — "
           "unlock it with the shared password at /login",
    "model": "the model views load a checkpoint and run it forward — CPU beside a live training "
             "run — unlock them with the shared password at /login",
}
_FRAGMENT = {"job": "partials/locked.html", "model": "partials/model_locked.html"}


class UnlockRequired(HTTPException):
    """A 403 that knows which fragment an HTMX target should show instead (see `app._unlock_required`)."""

    def __init__(self, kind: str, next_url: str) -> None:
        super().__init__(status_code=403, detail=_DETAIL[kind])
        self.kind = kind
        self.next_url = next_url
        self.template = _FRAGMENT[kind]


def unlock_gate(kind: str, next_url: str = "/") -> Any:
    """`Depends(...)` for `dependencies=[...]`: refuse (403) unless the request's cookie is unlocked."""
    if kind not in _DETAIL:
        raise ValueError(f"unknown gate kind {kind!r}")

    def _gate(request: Request) -> None:
        if not request.app.state.auth.unlocked(request.cookies.get(COOKIE)):
            raise UnlockRequired(kind, next_url)

    setattr(_gate, GATE_ATTR, kind)
    return Depends(_gate)


def job_gate(next_url: str = "/") -> Any:
    return unlock_gate("job", next_url)


def model_gate(next_url: str = "/game") -> Any:
    return unlock_gate("model", next_url)


def route_gate_kind(route: Any) -> "str | None":
    """`"job"` / `"model"` when `route` carries an unlock gate, else None. Reads FastAPI's resolved
    dependency tree, so a gate added by any means (decorator, router, nested dependency) counts."""
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return None
    stack = list(dependant.dependencies)
    while stack:
        d = stack.pop()
        kind = getattr(d.call, GATE_ATTR, None)
        if kind:
            return kind
        stack.extend(d.dependencies)
    return None
