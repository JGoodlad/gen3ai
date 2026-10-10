"""THE CLASS GUARD for the unlock gate (`web/gate.py`): every route that loads a checkpoint, runs the
model forward or starts a background job is gated behind the shared password.

WHY THIS EXISTS. `auth.py`'s rule is "reading needs nothing, WORK needs the password": an anonymous
endpoint that loads a checkpoint is a free CPU-burn button beside a live training run. `/game`'s model
endpoints shipped tagged "read-only" and ungated (the page at prober.g5d.io was public), and
`/api/analyze` had been ungated since the web app was written — two instances of one class, so the guard
is for the class, not for those routes.

THE MECHANISM, in two legs that check each other.

  1. STATIC, DERIVED FROM THE CODE (no hand-kept route list). The set of `ProbeSession` methods that
     reach model-loading code is computed by AST over `session/*.py`: a method is model-reaching if it
     calls `_model_for` / `ProbeModel` / `sanitized_load_custom_objects` / `capture_battle`, or calls
     (through `self.`) a method that is — transitively, so a new method built on `battle_readout`
     joins the set without anyone editing this file. Each route handler is then walked (its AST, its
     lambdas, and the app-level helper functions it calls): a handler that calls a model-reaching
     method, or `jobs.submit`, MUST carry the gate in the route table (`gate.route_gate_kind`), and a
     gated route MUST be one the derivation finds expensive, so the gate set cannot drift into
     decoration either. A NEW route is covered the moment it exists, because the route table is the
     input.
  2. DYNAMIC, BEHAVIOURAL. Every route is requested ANONYMOUSLY with the model-loading seam
     (`ProbeSession._model_for`) and `JobRegistry.submit` replaced by recorders: nothing may be
     recorded and every route the derivation named must answer "locked". Then the same sweep UNLOCKED:
     each of those routes must reach a recorder. That second half is what stops the first from being
     vacuous (a sweep that never reached the seam would pass on any code).

Both legs were proven to FAIL when a gate is removed from a route (`game.py`'s readout) and when
`_model_for` gains a new caller outside the gate.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from main.prober.web import fixture_run
from main.prober.web.app import create_app
from main.prober.web.gate import route_gate_kind

PASSWORD = "test-only-password"
_SESSION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "session")

#: what makes a session method model-reaching: a name it calls or refers to.
_SEEDS = {"_model_for", "ProbeModel", "sanitized_load_custom_objects", "capture_battle", "_capture"}


# -- leg 1: derive the expensive set from the code ----------------------------------------------

def _session_methods() -> "dict[str, tuple[set[str], set[str]]]":
    """method name -> (names it mentions, `self.X` names it calls), over every class in `session/`."""
    out: dict = {}
    for fn in sorted(os.listdir(_SESSION_DIR)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_SESSION_DIR, fn)).read())
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for m in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                mentions: set = set()
                calls: set = set()
                for n in ast.walk(m):
                    if isinstance(n, ast.Name):
                        mentions.add(n.id)
                    elif isinstance(n, ast.Attribute):
                        mentions.add(n.attr)
                        if isinstance(n.value, ast.Name) and n.value.id == "self":
                            calls.add(n.attr)
                    elif isinstance(n, (ast.Import, ast.ImportFrom)):
                        mentions.update(a.name.split(".")[-1] for a in n.names)
                old = out.get(m.name, (set(), set()))
                out[m.name] = (old[0] | mentions, old[1] | calls)
    return out


def model_reaching_methods() -> "set[str]":
    """Every `ProbeSession` method that (transitively) loads a checkpoint or runs the model."""
    methods = _session_methods()
    reach = {name for name, (mentions, _) in methods.items() if mentions & _SEEDS}
    changed = True
    while changed:
        changed = False
        for name, (_, calls) in methods.items():
            if name not in reach and calls & reach:
                reach.add(name)
                changed = True
    return reach


def _called_names(fn, seen: "set | None" = None) -> "set[str]":
    """Attribute names called anywhere in `fn` (lambdas included), following the web package's own
    helper functions it calls (closure or module globals) so a handler cannot hide a call in one."""
    seen = set() if seen is None else seen
    if fn in seen:
        return set()
    seen.add(fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return set()
    names = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    try:
        cv = inspect.getclosurevars(fn)
    except (TypeError, ValueError):
        return names
    for obj in list(cv.nonlocals.values()) + list(cv.globals.values()):
        if inspect.isfunction(obj) and obj.__module__.startswith("main.prober.web"):
            names |= _called_names(obj, seen)
        elif hasattr(obj, "__dataclass_fields__"):            # GameHelpers: its callables are the app's
            for v in vars(obj).values():
                if inspect.isfunction(v) and v.__module__.startswith("main.prober.web"):
                    names |= _called_names(v, seen)
    return names


def _api_routes(app) -> "list[APIRoute]":
    return [r for r in app.routes if isinstance(r, APIRoute)]


def expensive_kind(route: APIRoute, model_methods: "set[str]") -> "str | None":
    """What the handler does, derived from its code: "model" (reaches model-loading), "job"
    (submits a background job) or None."""
    called = _called_names(route.endpoint)
    # a handler wired through GameHelpers reaches the session through `h.session`, whose call sites
    # are the handler's own `sess.<method>(...)` — already in `called`.
    if called & model_methods:
        return "model"
    if "submit" in called:
        return "job"
    return None


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return fixture_run.build(str(tmp_path_factory.mktemp("proberweb_gate")))


@pytest.fixture(scope="module")
def app(run):
    return create_app(run, password=PASSWORD)


def test_the_derivation_finds_the_model_reaching_methods():
    """Non-vacuity of leg 1: a derivation that found nothing would pass everything. These are examples
    of what it must find, not the source of truth for what is gated."""
    found = model_reaching_methods()
    for must in ("analyze", "battle_readout", "decision_attention", "lookahead", "better_line",
                 "probe_model", "find"):
        assert must in found, (must, sorted(found))
    for model_free in ("battle_story", "battle_turns", "run_summary", "battles", "scan", "triage"):
        assert model_free not in found, f"{model_free} is model-free and must stay open"


def test_every_route_that_reaches_model_code_or_starts_a_job_is_gated(app):
    model_methods = model_reaching_methods()
    ungated = []
    for r in _api_routes(app):
        kind = expensive_kind(r, model_methods)
        if kind and route_gate_kind(r) is None:
            ungated.append(f"{sorted(r.methods)} {r.path}  ({kind})")
    assert not ungated, ("these routes load a checkpoint, run the model or start a job and are NOT "
                         "behind the unlock gate (add `dependencies=[model_gate(...)]` / "
                         "`job_gate(...)` from web/gate.py):\n  " + "\n  ".join(ungated))


def test_a_gated_route_is_one_the_code_says_is_expensive(app):
    """The converse: the gate is not decoration. A route gated but no longer reaching expensive code
    is either over-gated or its derivation broke — say so, rather than let the sets drift apart."""
    model_methods = model_reaching_methods()
    decor = [f"{sorted(r.methods)} {r.path}" for r in _api_routes(app)
             if route_gate_kind(r) and not expensive_kind(r, model_methods)]
    assert not decor, "gated but not expensive by the derivation:\n  " + "\n  ".join(decor)


def test_the_derived_set_covers_the_known_routes(app):
    """Leg 1 must actually SEE the routes this fix was written for (examples, not a list of record)."""
    model_methods = model_reaching_methods()
    got = {r.path: expensive_kind(r, model_methods) for r in _api_routes(app)}
    for path in ("/api/game/readout", "/api/game/attention", "/partials/game/model",
                 "/partials/game/attention", "/api/analyze", "/partials/analyze"):
        assert got.get(path) == "model", (path, got.get(path))
    # a job whose lambda calls a model-reaching method is "model" (the stronger statement); one that
    # only spawns re-rolls is "job". Either way it must be gated.
    assert got.get("/partials/job/lookahead") == "model"
    assert got.get("/api/jobs/falsify-scan") in ("job", "model")
    assert got.get("/api/game/story") is None and got.get("/game") is None


# -- leg 2: behaviour --------------------------------------------------------------------------

def _request(client, route: APIRoute, battle: str):
    path = route.path
    for name in ("job_id",):
        path = path.replace("{" + name + "}", "nope")
    q = {"battle": battle, "inv": "0", "action": "0"}
    body = {"battle": battle, "inv": "0", "action": "0"}
    method = sorted(route.methods - {"HEAD", "OPTIONS"})[0]
    if method == "GET":
        return client.get(path, params=q)
    return client.post(path, params=q, data=body)


class _Recorder:
    def __init__(self) -> None:
        self.hits: list = []


@pytest.fixture()
def swept(app, monkeypatch):
    """Anonymous client with the model-loading seam and job submission replaced by recorders."""
    from main.prober.session import ProbeSession
    rec = _Recorder()

    def boom_model(self, battle):
        rec.hits.append("model")
        raise FileNotFoundError("recorder: model seam reached")

    class _Jobs:
        def submit(self, kind, params, fn):
            rec.hits.append("job")
            raise RuntimeError("recorder: job submitted")

        def __getattr__(self, name):          # latest / get / list — the read side
            return getattr(real_jobs, name)

    real_jobs = app.state.jobs
    monkeypatch.setattr(ProbeSession, "_model_for", boom_model)
    monkeypatch.setattr(app.state, "jobs", _Jobs())
    with TestClient(app, raise_server_exceptions=False) as c:
        c.rec = rec
        yield c


def _skip(route: APIRoute) -> bool:
    return route.path in ("/login", "/logout")


def _battle(client) -> str:
    return client.get("/api/battles").json()[0]["short_id"]


def test_an_anonymous_sweep_of_every_route_never_reaches_the_model_or_the_job_pool(app, swept):
    battle = _battle(swept)
    model_methods = model_reaching_methods()
    leaked = []
    for r in _api_routes(app):
        if _skip(r):
            continue
        swept.rec.hits.clear()
        resp = _request(swept, r, battle)
        if swept.rec.hits:
            leaked.append(f"{sorted(r.methods)} {r.path} reached {swept.rec.hits} anonymously "
                          f"(HTTP {resp.status_code})")
        if expensive_kind(r, model_methods):
            if r.path.startswith("/partials/"):
                assert resp.status_code == 200 and "unlock" in resp.text.lower(), (r.path, resp.status_code)
            else:
                assert resp.status_code == 403, (r.path, resp.status_code, resp.text[:200])
    assert not leaked, "\n".join(leaked)


def test_the_same_sweep_unlocked_does_reach_the_seam(app, swept):
    """The other half of non-vacuity: with the cookie, each expensive route DOES reach a recorder, so
    the anonymous sweep above had something to catch."""
    r = swept.post("/login", data={"password": PASSWORD, "next": "/"}, follow_redirects=False)
    assert r.status_code == 303
    battle = _battle(swept)
    model_methods = model_reaching_methods()
    silent = []
    for route in _api_routes(app):
        kind = expensive_kind(route, model_methods)
        if not kind or _skip(route):
            continue
        swept.rec.hits.clear()
        _request(swept, route, battle)
        if not swept.rec.hits:
            silent.append(f"{sorted(route.methods)} {route.path} ({kind})")
    assert not silent, "unlocked requests that never reached the seam (the sweep is not exercising them):\n  " \
                       + "\n  ".join(silent)
