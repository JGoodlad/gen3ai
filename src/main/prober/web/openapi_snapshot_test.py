"""`openapi.json` is a COMMITTED contract, and this is its drift gate.

WHY SNAPSHOT IT. The reason FastAPI was chosen over a bare `http.server` was that it can emit a
machine-readable description of its own surface. That is only worth anything if the description is
checked in and compared — an OpenAPI document regenerated on demand describes whatever the code
happens to do today, which is not a contract, it is an echo. Committing it makes a route rename, a
dropped query parameter or a changed status code show up as a reviewable diff, exactly like
`delivery_graph_snapshot.json`.

The document is generated from `create_app(None)` — an app with no run directory — so the contract
can never vary with what happens to be in someone's `models/`.

Regenerate deliberately, with the same command the failure message prints:

    python -m main.prober.web --openapi
"""

from __future__ import annotations

import json
import os

from main.prober.web.__main__ import _DEFAULT_SNAPSHOT, _canonical, main, openapi_document


def test_the_committed_snapshot_is_current():
    want = _canonical(openapi_document())
    assert os.path.exists(_DEFAULT_SNAPSHOT), f"missing {_DEFAULT_SNAPSHOT}"
    with open(_DEFAULT_SNAPSHOT) as fh:
        have = fh.read()
    assert have == want, (
        "openapi.json is STALE — the API surface changed without the contract being updated.\n"
        "Regenerate with: python -m main.prober.web --openapi")


def test_check_openapi_exits_zero_when_current(capsys):
    assert main(["--check-openapi"]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_openapi_exits_one_on_drift(tmp_path, monkeypatch, capsys):
    """The gate has to actually FAIL on drift; a `--check` that always passes is worse than none.

    A stale file is simulated by pointing the module's snapshot path at a doctored copy, which is
    the same failure a renamed route would produce.
    """
    import main.prober.web.__main__ as M

    doc = openapi_document()
    doc["paths"]["/api/an-endpoint-that-does-not-exist"] = {"get": {"responses": {}}}
    stale = tmp_path / "openapi.json"
    stale.write_text(_canonical(doc))
    monkeypatch.setattr(M, "_DEFAULT_SNAPSHOT", str(stale))

    assert M.main(["--check-openapi"]) == 1
    assert "STALE" in capsys.readouterr().out


def test_check_openapi_reports_a_missing_snapshot(tmp_path, monkeypatch, capsys):
    import main.prober.web.__main__ as M
    monkeypatch.setattr(M, "_DEFAULT_SNAPSHOT", str(tmp_path / "gone.json"))
    assert M.main(["--check-openapi"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_the_contract_documents_every_route_the_app_serves():
    """Including the HTML pages. Hiding them would let a page be added, renamed or deleted without
    the gate noticing — which is the drift the gate exists for."""
    from main.prober.web.app import create_app

    doc = json.loads(_canonical(openapi_document()))
    served = {r.path for r in create_app(None).routes
              if getattr(r, "methods", None) and not r.path.startswith("/openapi")
              and r.path not in ("/docs", "/redoc", "/docs/oauth2-redirect")}
    assert served - {"/static/{path:path}"} <= set(doc["paths"])


def test_the_contract_pins_the_read_only_surface():
    doc = openapi_document()
    for path in ("/api/health", "/api/run", "/api/battles", "/api/scan", "/api/triage",
                 "/api/jobs", "/api/jobs/{job_id}", "/api/jobs/falsify-scan",
                 "/api/jobs/calibration", "/", "/battles", "/scan", "/triage", "/falsify",
                 "/calibration"):
        assert path in doc["paths"], f"{path} vanished from the contract"
    # Submitting a heavy probe must ACCEPT, not pretend to have finished it.
    assert "202" in doc["paths"]["/api/jobs/falsify-scan"]["post"]["responses"]


def test_query_parameters_are_documented_with_their_constraints():
    """The parameter list IS the contract for an agent driving this API."""
    doc = openapi_document()
    params = {p["name"]: p for p in doc["paths"]["/api/scan"]["get"]["parameters"]}
    assert set(params) == {"run", "outcome", "opponent", "step", "metric", "limit"}
    assert params["metric"]["schema"].get("default") == "value_drop"
    assert "value_drop|td_residual" in json.dumps(params["metric"]["schema"])


def test_every_read_route_takes_a_run_selector():
    """The picker is only real if each view can be asked for a specific run."""
    doc = openapi_document()
    for path in ("/api/run", "/api/battles", "/api/scan", "/api/triage",
                 "/", "/battles", "/scan", "/triage", "/falsify", "/calibration"):
        names = {p["name"] for p in doc["paths"][path]["get"].get("parameters", [])}
        assert "run" in names, f"{path} cannot be pointed at a run"


def test_the_auth_surface_is_in_the_contract():
    doc = openapi_document()
    for path in ("/login", "/logout", "/api/runs"):
        assert path in doc["paths"], f"{path} vanished from the contract"
    assert "post" in doc["paths"]["/login"]


def test_writing_the_snapshot_round_trips(tmp_path, capsys):
    out = tmp_path / "written.json"
    assert main(["--openapi", str(out)]) == 0
    assert json.loads(out.read_text())["info"]["title"] == "gen3ai prober"
    assert "wrote" in capsys.readouterr().out
