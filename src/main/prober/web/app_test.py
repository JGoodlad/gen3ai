"""The FastAPI adapter, over a synthetic run — no checkpoint, no bridge, no browser.

The claim under test is the one the whole package rests on: **the web view returns exactly what
`ProbeSession` returned**. So the API tests compare each endpoint against a direct session call on
the same run rather than against hand-written expected values — a reshaping bug in a handler then
fails here instead of quietly making the web numbers disagree with the CLI's.

The heavy probes are exercised with the session method REPLACED, because the point of those tests
is the job lifecycle and the rendering, not the re-roll machinery (which `falsifier_integration_test.py`
already owns and which needs Node).
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from main.prober.engine import BELIEF_NAME_CAVEAT
from main.prober.session import ProbeSession
from main.prober.web import fixture_run
from main.prober.web.app import create_app


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return fixture_run.build(str(tmp_path_factory.mktemp("proberweb")))


@pytest.fixture()
def client(run):
    """Pointed at the RUN itself, so the picker offers exactly that one — the pinned mode. The
    models-root/picker mode gets its own fixture below."""
    app = create_app(run, password="test-only-password")
    with TestClient(app) as c:
        c.app_state = app.state          # tests reach the sessions/registry through this
        yield c


def _the_session(client):
    """The one cached ProbeSession, whatever run path it is keyed on. Tests replace methods on it
    to stub the expensive probes."""
    client.get("/api/run")                                  # force it to exist
    (sess,) = client.app_state.sessions.values()
    return sess


def _unlock(client):
    """Exchange the shared password for the session cookie the job endpoints require."""
    r = client.post("/login", data={"password": "test-only-password", "next": "/falsify"},
                    follow_redirects=False)
    assert r.status_code == 303, r.text[:300]
    return r


# -- the JSON API mirrors ProbeSession exactly --------------------------------------------

def test_health_reports_the_root_and_the_lock_state(client, run):
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["models_root"] == run
    assert body["n_runs"] == 1, "pinned mode offers exactly the run it was pointed at"
    assert body["auth_required"] is True
    assert body["jobs_unlocked"] is False, "a fresh visitor has not unlocked anything"


def test_api_run_is_byte_for_byte_the_session_result(client, run):
    assert client.get("/api/run").json() == ProbeSession(run).run_summary()


def test_api_battles_matches_the_session_and_filters(client, run):
    sess = ProbeSession(run)
    assert client.get("/api/battles").json() == sess.battles()
    losses = client.get("/api/battles", params={"outcome": "loss"}).json()
    assert losses == sess.battles(outcome="loss")
    assert losses and all(b["outcome"] == "loss" for b in losses)
    assert len(losses) < len(sess.battles()), "the fixture must contain wins to exclude"


def test_api_battles_rejects_a_bogus_outcome(client):
    """The pattern is on the route, so a typo is a 422 rather than a silently unfiltered list."""
    assert client.get("/api/battles", params={"outcome": "drawn"}).status_code == 422


def test_api_scan_matches_the_session_and_honours_the_metric(client, run):
    sess = ProbeSession(run)
    assert client.get("/api/scan", params={"outcome": "loss"}).json() == sess.scan(outcome="loss")
    by_td = client.get("/api/scan", params={"outcome": "loss", "metric": "td_residual"}).json()
    assert by_td == sess.scan(outcome="loss", metric="td_residual")


def test_api_triage_matches_the_session(client, run):
    assert client.get("/api/triage").json() == ProbeSession(run).triage()


def test_the_battles_view_puts_the_sentinels_first(client, run):
    """Both halves the reader sees: the filter dropdown and the rows themselves. A sentinel game is
    the trainee against a recent SELF, so it says more about where the model is now than a fixed
    bot does — and with the 200-row cap an alphabetical accident could cut it entirely."""
    from main.prober.web.app import _by_opponent_strength, _opponents

    summary = ProbeSession(run).run_summary()
    names = _opponents(summary)
    assert names, "fixture has no opponents"

    # The fixture's opponents are bots, so the ORDERING rule is exercised on a synthetic list —
    # what this test pins on the real app is that both surfaces route through the shared key.
    ordered = _by_opponent_strength([
        {"opponent": "heuristic2"}, {"opponent": "sentinel_1"},
        {"opponent": "aggressive"}, {"opponent": "sentinel_0"}])
    assert [r["opponent"] for r in ordered][:2] == ["sentinel_0", "sentinel_1"]
    assert [r["opponent"] for r in ordered][2:] == ["heuristic2", "aggressive"], (
        "the bots were re-sorted; they should move as a block in their incoming order")

    assert _opponents({"steps": [{"opponents": [{"name": "staller"}, {"name": "sentinel_2"},
                                                {"name": "sentinel_0"}]}]}) == [
        "sentinel_0", "sentinel_2", "staller"]


def test_the_json_battles_endpoint_is_NOT_reordered(client, run):
    """Row order is a presentation choice about which rows a human meets first. `/api/battles`
    stays byte-identical to `ProbeSession.battles()` — a machine client asked for the run's
    battles, not for this page's opinion about them."""
    assert client.get("/api/battles").json() == ProbeSession(run).battles()


def test_api_awareness_matches_the_session(client, run):
    sess = ProbeSession(run)
    assert client.get("/api/awareness").json() == sess.awareness_scan()
    # An EMPTY outcome must reach the session as "every battle" — the quantile-coverage half of
    # this probe is only comparable to the published baseline unfiltered, so "all" cannot be an
    # unreachable option. (`""` is what an HTML select's "all" submits; the pattern allows it.)
    assert client.get("/api/awareness", params={"outcome": ""}).json() == \
        sess.awareness_scan(outcome=None)
    assert client.get("/api/awareness", params={"outcome": "drawn"}).status_code == 422


# -- errors are data, not 500s -------------------------------------------------------------

def test_an_api_404_is_a_parseable_error_envelope(client):
    """Mirrors the CLI's `{"error": ...}` contract, so an agent always gets JSON."""
    r = client.get("/api/nope")
    assert r.status_code == 404 and "error" in r.json()


def test_a_page_404_renders_html_not_json(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.text.lstrip().startswith("<!DOCTYPE html>")
    assert "gen3ai prober" in r.text


def test_no_models_directory_is_a_503_on_data_routes_but_health_still_answers():
    """`create_app(None)` is what `openapi.json` is generated from; it must be a live app."""
    with TestClient(create_app(None)) as c:
        assert c.get("/api/health").json()["n_runs"] == 0
        r = c.get("/api/run")
        assert r.status_code == 503 and "no models directory" in r.json()["error"]


def test_a_nonexistent_root_is_refused_at_construction(tmp_path):
    """`RunStore` will not build on a path that is not a directory, so a typo cannot become a
    confident "0 battles captured" page."""
    from main.prober.web.runs import RunAccessError
    with pytest.raises(RunAccessError):
        create_app(str(tmp_path / "does_not_exist"))


def test_the_cli_refuses_to_serve_a_path_that_does_not_exist(tmp_path, capsys):
    from main.prober.web.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "does_not_exist")])
    assert exc.value.code != 0
    assert "does_not_exist" in capsys.readouterr().err


# -- pages + HTMX fragments -----------------------------------------------------------------

@pytest.mark.parametrize("path,marker", [
    ("/", "Run summary"), ("/battles", "Battles"), ("/scan", "Turning-point scan"),
    ("/triage", "Loss-lever triage"), ("/falsify", "Crater bracket"),
    ("/calibration", "Critic calibration")])
def test_every_page_renders_a_document(client, path, marker):
    r = client.get(path)
    assert r.status_code == 200
    assert r.text.lstrip().startswith("<!DOCTYPE html>")
    assert marker in r.text
    assert 'src="/static/vendor/htmx.min.js"' in r.text
    assert "cdn.jsdelivr.net" not in r.text, "a CDN reference reintroduces the offline-skip trap"


def test_pages_reference_only_vendored_scripts(client):
    """Every `<script src>` must be local. This is the property that keeps the render test from
    degrading into a skip when the box has no network."""
    import re
    for path in ("/", "/battles", "/scan", "/triage", "/falsify", "/calibration"):
        for src in re.findall(r'<script src="([^"]+)"', client.get(path).text):
            assert src.startswith("/static/"), f"{path} loads a remote script: {src}"


def test_static_assets_are_actually_served(client):
    for path in ("/static/app.css", "/static/app.js", "/static/vendor/htmx.min.js",
                 "/static/vendor/vega.min.js", "/static/vendor/vega-lite.min.js",
                 "/static/vendor/vega-embed.min.js"):
        r = client.get(path)
        assert r.status_code == 200 and len(r.content) > 500, path


def test_run_page_embeds_a_chart_spec(client):
    r = client.get("/")
    assert 'class="chart" data-chart="outcomes"' in r.text
    assert "vega-lite/v5.json" in r.text


def test_battles_fragment_filters_and_counts(client, run):
    all_rows = client.get("/partials/battles").text
    losses = client.get("/partials/battles", params={"outcome": "loss"}).text
    n_all = ProbeSession(run).battles()
    n_loss = ProbeSession(run).battles(outcome="loss")
    assert f'data-count="{len(n_all)}"' in all_rows
    assert f'data-count="{len(n_loss)}"' in losses
    assert "<!DOCTYPE" not in losses, "a fragment must not carry the page shell"


def test_battles_fragment_says_so_when_nothing_matches(client):
    body = client.get("/partials/battles", params={"opponent": "nobody"}).text
    assert "no battles match" in body


# A `<select>` whose "all" option is `value=""` submits the field as an EMPTY STRING, and every
# numeric filter on these forms has one. Strictly-typed `int | None` params 422 on that — which is
# exactly what shipped, passed every unit test above (none of them sent the empty field), and was
# caught only by the headless browser. These send what the FORM sends.

@pytest.mark.parametrize("path", ["/partials/battles", "/partials/scan", "/partials/triage"])
def test_a_fragment_survives_the_empty_fields_a_real_form_submits(client, path):
    form = {"outcome": "loss", "opponent": "", "step": "", "metric": "value_drop", "limit": "25"}
    r = client.get(path, params=form)
    assert r.status_code == 200, f"{path} rejected a real form submission: {r.text[:300]}"
    assert "no battles match" not in r.text or path == "/partials/battles"


def test_the_empty_step_field_means_all_steps_not_an_error(client, run):
    """Blank must be 'no filter', not 'filter to nothing' — the two are indistinguishable in the
    rendered table, so this pins which one it is."""
    blank = client.get("/partials/battles", params={"step": ""}).text
    assert f'data-count="{len(ProbeSession(run).battles())}"' in blank


@pytest.mark.parametrize("path", ["/partials/job/falsify-scan", "/partials/job/calibration"])
def test_the_job_forms_also_survive_their_empty_fields(client, path):
    _unlock(client)
    _the_session(client).falsify_scan = lambda **kw: _FAKE_FALSIFY
    _the_session(client).calibration = lambda **kw: _FAKE_CALIBRATION
    r = client.post(path, params={"outcome": "loss", "opponent": "", "step": ""})
    assert r.status_code == 200, r.text[:300]
    assert "data-job-id" in r.text


def test_the_api_stays_strict_where_the_fragments_are_lenient(client):
    """The leniency is a concession to HTML forms, not a general loosening: a machine client
    passing a malformed int should be told so."""
    assert client.get("/api/battles", params={"step": ""}).status_code == 422
    assert client.get("/partials/battles", params={"step": ""}).status_code == 200


def test_scan_fragment_carries_the_table_and_its_chart(client, run):
    body = client.get("/partials/scan", params={"outcome": "loss"}).text
    worst = ProbeSession(run).scan(outcome="loss")[0]
    assert worst["short_id"] in body
    assert worst["worst"]["chosen"] in body
    assert 'data-chart="scan"' in body


def test_triage_fragment_shows_the_categories_and_the_caveats(client, run):
    body = client.get("/partials/triage").text
    data = ProbeSession(run).triage()
    assert data["categories"], "the fixture must produce at least one triage category"
    assert data["categories"][0]["category"] in body
    assert 'data-chart="triage"' in body
    # The caveats are the difference between a bound and a number on a dashboard.
    assert "caveats" in body and "UPPER BOUND" in body.upper()


# -- background jobs --------------------------------------------------------------------------

_FAKE_FALSIFY = {
    "gate": {"policy_reducible": 0.25, "aleatoric": 0.35, "unattributed": 0.3, "mixed": 0.1,
             "critic_headroom_upper_bound": 0.65},
    "weighting": "delta",
    "verdict_counts": {"LUCK": 3, "MISTAKE": 2, "NEUTRAL": 4, "MIXED": 1},
    "count_shares": {"LUCK": 0.3, "MISTAKE": 0.2, "NEUTRAL": 0.4, "MIXED": 0.1},
    "weighted_shares": {"LUCK": 0.35, "MISTAKE": 0.25, "NEUTRAL": 0.3, "MIXED": 0.1},
    "coverage": {"n_matched": 3, "n_with_record": 3, "n_falsified": 3, "n_capped_by_limit": 0,
                 "n_skipped_no_record": 0, "n_battle_errors": 0, "n_decisions": 10,
                 "n_decision_errors": 0, "skipped_no_record_sample": []},
    "dominant_lever": "unattributed",
    "interpretation": "a synthetic interpretation line",
    "caveats": ["a synthetic caveat"], "battles": [], "errors": [],
}

_FAKE_CALIBRATION = {
    "gate": {"policy_reducible": 0.2, "aleatoric": 0.3, "unattributed": 0.5,
             "unattributed_critic_overvalued": 0.2, "unattributed_lost_position": 0.3,
             "critic_mean_reducible_upper_bound": 0.2},
    "falsify_gate": {"policy_reducible": 0.2, "aleatoric": 0.3, "unattributed": 0.5,
                     "mixed": 0.0, "critic_headroom_upper_bound": 0.8},
    "overall_calibration": {"bias": 1.5, "bias_on_wins": -2.0, "bias_on_losses": 4.0,
                            "captured_win_fraction": 0.33, "mae": 5.0, "ev": 0.4,
                            "slope": 0.9, "n": 120},
    "reliability_curve": [{"v_lo": -9.0, "v_hi": -1.0, "v_mean": -5.0, "g_mean": -7.0,
                           "n": 60, "gap": 2.0},
                          {"v_lo": -1.0, "v_hi": 8.0, "v_mean": 3.0, "g_mean": 3.5,
                           "n": 60, "gap": -0.5}],
    "unattributed_resolution": {"n_unattributed_craters": 5, "n_without_gap": 1,
                                "overvalued_share_of_unattributed": 0.4,
                                "mean_reliability_gap": 1.1, "examples": []},
    "interpretation": "synthetic calibration interpretation",
    "caveats": ["SELECTION CONFOUND (dominant): synthetic"],
}


def _await_job(client, job_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["done"]:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def test_a_job_runs_off_the_request_and_its_result_renders(client):
    _unlock(client)
    _the_session(client).falsify_scan = lambda **kw: _FAKE_FALSIFY

    r = client.post("/api/jobs/falsify-scan", params={"limit": 3})
    assert r.status_code == 202, "submitting must ACCEPT and return, not block for minutes"
    job_id = r.json()["id"]

    body = _await_job(client, job_id)
    assert body["status"] == "done"
    assert body["result"]["gate"]["critic_headroom_upper_bound"] == 0.65

    html = client.get(f"/partials/job/{job_id}").text
    assert 'data-job-status="done"' in html
    assert 'data-chart="bracket"' in html, "the finished job must render its crater bracket"
    assert "a synthetic interpretation line" in html
    assert "a synthetic caveat" in html
    assert "hx-trigger=\"every 2s\"" not in html, "a finished job must stop polling the server"


@pytest.mark.parametrize("path,method,extra", [
    ("falsify-scan", "falsify_scan", {}),
    ("calibration", "calibration", {"bins": "4"}),
])
def test_the_page_form_fields_actually_reach_the_probe(client, path, method, extra):
    """REGRESSION, and it shipped broken: an HTMX `<form hx-post>` serializes its fields into the
    request BODY, but these handlers declared them as `Query(...)` — which reads the URL. So every
    control on /falsify and /calibration was silently ignored and the probe ran at its defaults.

    Measured before the fix: a submit of outcome=win/limit=3/seeds=7/concurrency=4 arrived as
    loss/20/32/1. `outcome` is the one that makes this a correctness bug rather than an
    inconvenience — asking for WINS quietly scanned LOSSES and returned a confident answer to a
    question nobody asked."""
    _unlock(client)
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return _FAKE_FALSIFY if method == "falsify_scan" else _FAKE_CALIBRATION

    setattr(_the_session(client), method, spy)
    client.post(f"/partials/job/{path}",
                data=dict({"outcome": "win", "limit": "3", "seeds": "7", "alts": "5",
                           "concurrency": "4", "opponent": "heuristic2", "worst": "6"}, **extra))
    for _ in range(100):
        if seen:
            break
        time.sleep(0.05)

    assert seen, "the job never ran"
    assert seen["outcome"] == "win"            # the dangerous one: not silently 'loss'
    assert seen["opponent"] == "heuristic2"
    assert seen["limit"] == 3 and seen["worst"] == 6
    assert seen["n_seeds"] == 7 and seen["n_alts"] == 5
    assert seen["concurrency"] == 4


def test_a_running_job_renders_a_poller_that_stops_when_done(client):
    _unlock(client)
    client.get("/api/run")
    gate = {"go": False}

    def slow(**kw):
        while not gate["go"]:
            time.sleep(0.01)
        return _FAKE_FALSIFY

    _the_session(client).falsify_scan = slow
    job_id = client.post("/api/jobs/falsify-scan").json()["id"]
    try:
        html = client.get(f"/partials/job/{job_id}").text
        assert 'hx-trigger="every 2s"' in html, "an unfinished job must keep polling"
        assert "spinner" in html
    finally:
        gate["go"] = True
    _await_job(client, job_id)


def test_a_failing_probe_is_a_rendered_message_not_a_500(client):
    """The expected outcome on a run with no `reconstruction.json` siblings — which is exactly
    what the fixture is. It must read as a message, not an error page."""
    _unlock(client)
    client.get("/api/run")

    def boom(**kw):
        raise RuntimeError("no *_reconstruction.json sibling; bridge-eval traces only")

    _the_session(client).calibration = boom
    job_id = client.post("/api/jobs/calibration").json()["id"]
    body = _await_job(client, job_id)
    assert body["status"] == "error"
    assert "reconstruction.json" in body["error"]

    html = client.get(f"/partials/job/{job_id}")
    assert html.status_code == 200
    assert "reconstruction.json" in html.text


def test_the_calibration_result_renders_its_curve_and_the_selection_diagnostics(client):
    _unlock(client)
    _the_session(client).calibration = lambda **kw: _FAKE_CALIBRATION
    job_id = client.post("/api/jobs/calibration").json()["id"]
    _await_job(client, job_id)

    html = client.get(f"/partials/job/{job_id}").text
    assert 'data-chart="reliability"' in html and 'data-chart="gap"' in html
    assert 'data-chart="bracket"' in html
    # The win/loss bias split is the finding; the headline alone is the misreading.
    assert "bias on wins" in html and "bias on losses" in html
    assert "SELECTION CONFOUND" in html


def test_a_page_restores_the_latest_job_after_a_reload(client):
    _unlock(client)
    _the_session(client).falsify_scan = lambda **kw: _FAKE_FALSIFY
    job_id = client.post("/api/jobs/falsify-scan").json()["id"]
    _await_job(client, job_id)
    assert f"/partials/job/{job_id}" in client.get("/falsify").text


def test_an_unknown_job_id_is_a_404_on_both_surfaces(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/partials/job/nope").status_code == 404


def test_the_job_list_omits_results(client):
    _unlock(client)
    _the_session(client).falsify_scan = lambda **kw: _FAKE_FALSIFY
    job_id = client.post("/api/jobs/falsify-scan").json()["id"]
    _await_job(client, job_id)
    rows = client.get("/api/jobs").json()
    row = next(r for r in rows if r["id"] == job_id)
    assert "result" not in row, "the index must stay small; results are fetched per id"


def test_probe_session_is_built_once_per_run_and_reused(client):
    """Constructing it walks the trace tree; a per-request session would re-walk on every page."""
    first = _the_session(client)
    client.get("/api/battles")
    client.get("/scan")
    assert _the_session(client) is first
    assert len(client.app_state.sessions) == 1


# -- the password gate, at the HTTP layer -------------------------------------------------
# The unit-level properties live in `auth_test.py`. What matters here is the POLICY: reading is
# anonymous, spending CPU is not, and a locked visitor is told how to proceed rather than being
# handed a bare 403.

@pytest.mark.parametrize("path", ["/", "/battles", "/scan", "/triage", "/falsify", "/calibration",
                                  "/api/run", "/api/battles", "/api/scan", "/api/triage",
                                  "/api/runs", "/api/health", "/api/jobs"])
def test_every_read_view_is_anonymous(client, path):
    """The model is open source and its outcomes are meant to be public. Nothing on the reading
    side may ever ask for the password."""
    assert client.get(path).status_code == 200, path


@pytest.mark.parametrize("path", ["/api/jobs/falsify-scan", "/api/jobs/calibration"])
def test_the_job_api_is_403_without_the_password(client, path):
    r = client.post(path)
    assert r.status_code == 403
    assert "/login" in r.json()["error"], "the refusal must say how to proceed"


@pytest.mark.parametrize("path", ["/partials/job/falsify-scan", "/partials/job/calibration"])
def test_the_job_form_renders_a_way_in_rather_than_an_error(client, path):
    """HTMX would swallow a 403 into the generic error handler, so the locked state is a rendered
    fragment with a link — not a status code the page cannot explain."""
    r = client.post(path)
    assert r.status_code == 200
    assert 'data-job-status="locked"' in r.text
    assert "/login" in r.text
    assert "shared password" in r.text


def test_the_password_unlocks_the_job_endpoints(client):
    assert client.post("/api/jobs/falsify-scan").status_code == 403
    _unlock(client)
    _the_session(client).falsify_scan = lambda **kw: _FAKE_FALSIFY
    assert client.post("/api/jobs/falsify-scan").status_code == 202
    assert client.get("/api/health").json()["jobs_unlocked"] is True


def test_the_wrong_password_does_not_unlock(client):
    r = client.post("/login", data={"password": "pidgey", "next": "/falsify"})
    assert "not the password" in r.text
    assert client.post("/api/jobs/falsify-scan").status_code == 403


def test_logout_relocks(client):
    _unlock(client)
    assert client.get("/api/health").json()["jobs_unlocked"] is True
    client.post("/logout")
    assert client.get("/api/health").json()["jobs_unlocked"] is False


def test_the_session_cookie_is_httponly_and_not_the_password(client):
    """A cookie is client-visible and lands in proxy logs; it must be a signature, never the
    secret itself."""
    r = _unlock(client)
    raw = r.headers["set-cookie"]
    assert "httponly" in raw.lower()
    assert "samesite=lax" in raw.lower(), "Lax is the CSRF story for the job POSTs"
    assert "test-only-password" not in raw


def test_login_will_not_redirect_off_site(client):
    """An open redirect on a login form is the classic phishing primitive."""
    for evil in ("https://evil.example", "//evil.example", "/\\evil"):
        r = client.post("/login", data={"password": "test-only-password", "next": evil},
                        follow_redirects=False)
        assert r.headers["location"] == "/", evil


def test_with_open_access_no_password_is_needed(run):
    """`--open` is the laptop mode."""
    app = create_app(run, open_access=True)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["auth_required"] is False
        c.get("/api/run")
        (sess,) = app.state.sessions.values()
        sess.falsify_scan = lambda **kw: _FAKE_FALSIFY
        assert c.post("/api/jobs/falsify-scan").status_code == 202


def test_with_no_password_configured_the_probes_are_off(run):
    """Fails CLOSED: an operator who forgets the secret publishes a read-only site, not a
    CPU-burn button."""
    with TestClient(create_app(run)) as c:
        assert c.get("/api/run").status_code == 200
        assert c.post("/api/jobs/falsify-scan").status_code == 403
        body = c.get("/login").text.lower()
        assert "no password set" in body, "the login page must explain why it cannot help"
        assert "disabled" in body, "and not offer a form that could never work"


# -- the run picker ------------------------------------------------------------------------

@pytest.fixture()
def models_client(tmp_path_factory):
    """Picker mode: a models ROOT holding two runs."""
    import shutil
    root = tmp_path_factory.mktemp("models")
    fixture_run.build(str(root))                        # -> <root>/run_fixture
    staging = tmp_path_factory.mktemp("staging")
    second = fixture_run.build(str(staging))
    shutil.move(second, str(root / "run_other"))        # a second, distinct run
    app = create_app(str(root), password="test-only-password")
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def test_the_picker_lists_every_run(models_client):
    names = {r["name"] for r in models_client.get("/api/runs").json()}
    assert names == {"run_fixture", "run_other"}


def test_the_picker_appears_in_the_page_shell(models_client):
    html = models_client.get("/").text
    assert 'name="run"' in html and "run_other" in html and "run_fixture" in html


def test_each_run_gets_its_own_session(models_client):
    models_client.get("/api/run", params={"run": "run_fixture"})
    models_client.get("/api/run", params={"run": "run_other"})
    assert len(models_client.app_state.sessions) == 2, "runs must not share a ProbeSession"


def test_an_unknown_run_is_a_404_that_does_not_echo_the_input(models_client):
    r = models_client.get("/api/run", params={"run": "../../etc/passwd"})
    assert r.status_code == 404
    assert "etc" not in r.json()["error"]


def test_the_run_choice_survives_into_the_fragments(models_client):
    """The picker is useless if the HTMX tables keep showing the default run."""
    body = models_client.get("/partials/battles", params={"run": "run_other"})
    assert body.status_code == 200 and "data-count=" in body.text


def test_page_links_carry_the_selected_run(models_client):
    html = models_client.get("/", params={"run": "run_other"}).text
    assert "?run=run_other" in html, "nav links must keep you on the run you picked"


# -- regressions from the 2026-08-09 adversarial review --------------------------------------

def test_a_spoofed_forwarding_header_is_ignored_from_an_untrusted_peer(client):
    """The review's headline finding: `_client` trusted CF-Connecting-IP unconditionally, so
    rotating it per request defeated the throttle entirely (500/500 guesses accepted).

    TestClient's peer is "testclient", which is NOT in the trusted set — so the header must be
    ignored and every one of these attempts must share one identity and hit the cooldown."""
    accepted = 0
    for i in range(40):
        r = client.post("/login", data={"password": "wrong", "next": "/falsify"},
                        headers={"CF-Connecting-IP": f"10.9.0.{i}"})
        if "Too many attempts" in r.text:
            break
        accepted += 1
    assert accepted < 40, "a rotating CF-Connecting-IP still bypassed the throttle"


def test_x_forwarded_for_is_never_honoured(client):
    """It is the append-friendly header, so a client-supplied value can survive even behind a
    proxy. It was removed entirely rather than trust-gated."""
    from main.prober.web.app import _client, _TRUSTED_PEERS
    from starlette.datastructures import Headers

    class _Req:
        headers = Headers({"x-forwarded-for": "1.2.3.4", "cf-connecting-ip": "5.6.7.8"})
        class client:  # noqa: D106
            host = "127.0.0.1"
    assert "127.0.0.1" in _TRUSTED_PEERS
    assert _client(_Req()) == "5.6.7.8", "the trusted header should win"

    class _Untrusted(_Req):
        class client:  # noqa: D106
            host = "203.0.113.9"
    assert _client(_Untrusted()) == "203.0.113.9", "an untrusted peer's header must be ignored"


def test_a_pinned_run_with_an_awkward_name_still_opens(tmp_path):
    """`list_runs()` advertised a pinned run whose basename failed the enumeration pattern while
    `resolve()` refused it — the app 404'd on its own and only run."""
    from main.prober.web import fixture_run
    from main.prober.web.runs import RunStore

    built = fixture_run.build(str(tmp_path))
    awkward = tmp_path / "my run (v8)"
    import shutil
    shutil.move(built, str(awkward))

    store = RunStore(str(awkward))
    name = store.default_run()
    assert name == "my run (v8)"
    assert store.resolve(name), "the pinned run must open under its own advertised name"

    with TestClient(create_app(str(awkward))) as c:
        assert c.get("/api/run").status_code == 200
        assert c.get("/").status_code == 200


# -- usability / information flow -------------------------------------------------------------
# The complaint these answer: "the old TUI was hard to work with because the information didn't
# flow at all." Each test pins one link in the chain from question -> answer -> next step.

def test_the_nav_follows_the_documented_investigation_recipe(client):
    """src/main/prober/CLAUDE.md says of triage: "start here for 'what next'". Six equal tabs in
    an arbitrary order is the flow complaint restated."""
    from main.prober.web.app import _NAV
    labels = [label for _, label in _NAV]
    assert labels.index("triage") < labels.index("scan") < labels.index("battles")
    assert labels.index("falsify") < labels.index("calibration")


def test_the_landing_page_tells_you_where_to_start(client):
    """A visitor with a question must not have to guess which of six tabs holds it."""
    html = client.get("/").text
    assert "Where to start" in html
    for question in ("Which failure lever", "Where exactly did each loss go wrong",
                     "bad luck or a reducible mistake"):
        assert question in html, question
    start = html.index("Where to start")
    assert html.index("/triage", start) < html.index("/calibration", start), (
        "the start list must be in recipe order")


def test_every_page_names_the_run_it_is_showing(client, run):
    """Otherwise the run being viewed exists only inside a dropdown widget, and a screenshot of a
    chart carries no record of where it came from."""
    import os
    name = os.path.basename(run)
    for path in ("/", "/battles", "/scan", "/triage", "/falsify", "/calibration"):
        html = client.get(path).text
        assert "ctxstrip" in html, f"{path} has no context strip"
        assert name in html, f"{path} does not name its run"
        assert "eval steps" in html and "battles captured" in html


def test_battles_and_triage_arrive_populated_rather_than_empty(client):
    """Measured: battles 119 ms, triage 436 ms — cheap enough that landing on an empty page and
    waiting for a round trip is a pure loss. (Scan stays async at ~2 s; see below.)"""
    battles = client.get("/battles").text
    assert "<table" in battles and "data-row" in battles
    assert 'hx-trigger="change"' in battles and "load" not in battles.split("hx-trigger")[1][:30]

    triage = client.get("/triage").text
    assert "<table" in triage and "data-row" in triage
    assert 'data-chart="triage"' in triage


def test_scan_stays_async_but_says_what_it_is_doing(client):
    """2 s of white page is worse than a page that arrives and fills in — but "loading…" tells the
    reader nothing about why."""
    html = client.get("/scan").text
    assert "<table" not in html, "scan should not block the first paint"
    assert "every captured battle" in html, "the waiting state must explain the wait"


def test_a_scan_row_can_be_taken_onward(client):
    """THE biggest flow break: the table names the exact decision that lost a battle and used to
    offer no way to act on it. `analyze` is deliberately not a web view, so the honest handoff is
    the command."""
    html = client.get("/partials/scan", params={"outcome": "loss"}).text
    assert "data-battle-id=" in html
    assert "copybtn" in html
    assert "main.prober.query analyze" in html, "no copyable command to continue the investigation"


def test_the_cryptic_scan_columns_explain_themselves(client):
    """`inv`, `ΔV` and `TD δ` are precise and meaningless to anyone who has not read engine.py."""
    html = client.get("/partials/scan", params={"outcome": "loss"}).text
    assert "Invocation index" in html
    assert "how far the critic" in html.replace("&#39;", "'")
    assert "TD residual" in html


def test_the_battles_table_is_capped_and_says_so(client):
    """2397 rows rendered to 545 KB is a download, not a table."""
    from main.prober.web.app import _BATTLE_PAGE
    assert _BATTLE_PAGE <= 500
    html = client.get("/partials/battles").text
    assert "data-count=" in html


def test_the_run_picker_is_grouped(models_client):
    """79 flat near-identical names is a scanning task; generations are the natural grouping."""
    html = models_client.get("/").text
    assert "<optgroup" in html


def test_grouping_buckets_by_generation():
    from main.prober.web.app import _group_runs
    rows = [{"name": "ai_v9_06_gen5_x"}, {"name": "ai_v9_05_gen4_y"}, {"name": "ai_v8_13_z"},
            {"name": "run_20260808_212910_gen4"}, {"name": "oddity"}]
    groups = dict(_group_runs(rows))
    assert len(groups["ai_v9"]) == 2
    assert len(groups["ai_v8"]) == 1
    assert "run 2026-08" in groups
    assert "other" in groups


def test_a_v_only_triage_warns_that_the_split_is_skewed(client):
    """`win-prob coverage 0%` as a bare stat is a number nobody can act on. It actually means the
    winning/behind split fell back to V > 0, which the project's own docs call systematically
    wrong — so it belongs in prose, next to the two categories it distorts."""
    html = client.get("/partials/triage").text
    if "winning/behind signal" in html and "V only" in html:
        assert "no win-prob head" in html
        assert "over-counts" in html


def test_the_probe_engine_is_a_startup_choice_threaded_into_every_session(run):
    """`ProbeSession` treats `impl` as SESSION-WIDE ("two probes of one run answered under
    different engines would not be comparable"), so the web surfaces it as a startup flag rather
    than a query param — and it must actually reach the sessions it builds, not just be stored."""
    app = create_app(run, impl="rust")
    with TestClient(app) as c:
        assert c.get("/api/health").json()["impl"] == "rust"
        c.get("/api/run")
        (sess,) = app.state.sessions.values()
        assert sess._impl == "rust", "the engine choice never reached ProbeSession"


def test_the_probe_engine_defaults_to_node(client):
    """Unchanged behaviour unless someone asks for rust."""
    assert client.get("/api/health").json()["impl"] == "node"
    assert _the_session(client)._impl == "node"


def test_the_session_cache_is_bounded_and_closes_what_it_evicts(tmp_path_factory):
    """A `scan` of one run leaves ~430 MB of cached summaries behind (measured on the real
    models/: 6 runs -> 3.0 GB, monotonic), and the picker offers 81 runs. Unbounded, that is an
    anonymous visitor's lever on ~35 GB of a box that is training — the same defect class as the
    auth failure map.
    """
    import shutil
    from main.prober.web.app import _MAX_CACHED_SESSIONS

    root = tmp_path_factory.mktemp("many")
    staging = tmp_path_factory.mktemp("staging")
    names = []
    for i in range(_MAX_CACHED_SESSIONS + 2):
        built = fixture_run.build(str(staging / f"s{i}"))
        dest = root / f"run_{i}"
        shutil.move(built, str(dest))
        names.append(dest.name)

    app = create_app(str(root))
    closed = []
    with TestClient(app) as c:
        for name in names:
            r = c.get("/api/run", params={"run": name})
            assert r.status_code == 200, name
            # Record closes on whatever is currently cached, so an eviction is observable.
            for sess in app.state.sessions.values():
                if not hasattr(sess, "_close_spy"):
                    sess._close_spy = True
                    original = sess.close
                    sess.close = lambda o=original, s=sess: (closed.append(s), o())[1]

        assert len(app.state.sessions) <= _MAX_CACHED_SESSIONS, (
            f"cache grew to {len(app.state.sessions)} — an anonymous visitor can walk every run")
        assert closed, "evicted sessions must be closed, not left for the collector"


# -- /battle: the turn-by-turn replay ---------------------------------------------------------
#
# The security tests here are the important ones. `/battle` is the first view that takes a
# per-TRACE identifier from the client, and `ProbeSession._battle` falls back to opening an
# arbitrary path for an id it does not recognise — so this endpoint is exactly where the run
# picker's "membership, not sanitisation" rule has to be repeated one level down.

_REPLAY_BATTLE = "step_4000000/heuristic2/loss_003"       # the fixture battle with a real log


def test_api_battle_turns_is_byte_for_byte_the_session_result(client, run):
    got = client.get("/api/battle-turns", params={"battle": _REPLAY_BATTLE}).json()
    want = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    assert got == want


def test_api_battle_turns_defaults_to_a_battle_rather_than_erroring(client, run):
    body = client.get("/api/battle-turns").json()
    assert body["short_id"] in {b["short_id"] for b in ProbeSession(run).battles()}


def test_an_unknown_battle_is_a_404_that_does_not_echo_the_input(client):
    r = client.get("/api/battle-turns", params={"battle": "step_1/nope/loss_999"})
    assert r.status_code == 404
    assert "nope" not in r.json()["error"]


@pytest.mark.parametrize("attack", [
    "../../../etc/passwd",
    "/etc/passwd",
    "%2e%2e%2fetc%2fpasswd",
    "step_4000000/heuristic2/../../../loss_003",
    "step_4000000/heuristic2/loss_003\x00",
])
def test_a_battle_token_is_never_joined_to_a_path(client, attack):
    """The run picker's rule, one level down: the token is only ever tested for MEMBERSHIP in the
    server's own listing. A traversal string cannot appear in that listing, so it cannot select
    anything — nothing here is being sanitised, and that is the point."""
    r = client.get("/api/battle-turns", params={"battle": attack})
    assert r.status_code == 404
    assert "passwd" not in r.text and "etc" not in r.json().get("error", "")


def test_a_summary_path_from_another_run_is_refused(tmp_path_factory):
    """The concrete escape this guards. `ProbeSession._battle` accepts a raw `*_summary.json` PATH
    and will happily `build_trace_tree` one belonging to a different run. Serving a single pinned
    run must not become a way to read its siblings' traces."""
    import shutil
    root = tmp_path_factory.mktemp("two")
    pinned = fixture_run.build(str(root))                      # -> <root>/run_fixture
    staging = tmp_path_factory.mktemp("stage")
    other = shutil.move(fixture_run.build(str(staging)), str(root / "run_secret"))
    leaked = os.path.join(other, "eval_traces", "step_4000000", "heuristic2",
                          "loss_003_summary.json")
    assert os.path.exists(leaked)                              # the attack is well-formed

    app = create_app(pinned)                                   # pinned to ONE run
    with TestClient(app) as c:
        r = c.get("/api/battle-turns", params={"battle": leaked})
        assert r.status_code == 404, "a path from another run resolved — the token reached a path"
        assert "run_secret" not in r.text


def test_the_battle_page_renders_the_turns_with_their_log(client):
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "Battle replay" in html
    assert html.count('class="card turn"') == 2, "one card per game turn"
    # The damage is re-attributed: our icebeam's number is the OPPONENT's recorded HP loss (-31%),
    # which is the misreading `build_result_timeline` exists to fix.
    assert "we icebeam did 31%" in html, "the battle log is the whole point of this view"
    assert "opp earthquake did 22%" in html
    assert "hpfill" in html, "an HP bar is sized from the session's hp_pct"
    assert 'id="inv-0"' in html, "each decision needs an anchor to be linkable"


def test_the_replay_says_when_the_move_order_is_unknown(client):
    """The fixture records no `move_order`, so top-to-bottom is NOT the real sequence. Showing an
    implied order we cannot ground in the log would be a quiet lie — but the caveat is a legend
    printed ONCE plus a marker per log, not the full sentence repeated under all fifty turns."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert "log unordered" in html, "the dashed marker is the whole signal"
    assert "order not recorded" in html
    assert html.count("not necessarily in the order shown") == 1, (
        "the caveat should be explained once, not repeated under every turn")


def test_the_replay_window_pages_and_deep_links(client, run):
    """A 249-turn battle (measured, real run) is a download, not a page — so it is windowed, and
    every position in it has to be a URL you can send someone."""
    from main.prober.web.app import _TURN_PAGE
    assert _TURN_PAGE <= 100
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE, "start": "2"}).text
    assert 'data-turn="2"' in html
    assert 'data-turn="1"' not in html, "start= must window the replay, not just scroll it"


def test_a_bad_start_value_does_not_break_the_page(client):
    """A hand-edited URL is a browser control by another name: it must never 422 the view."""
    for start in ("", "abc", "-5", "99999"):
        r = client.get("/battle", params={"battle": _REPLAY_BATTLE, "start": start})
        assert r.status_code == 200, f"start={start!r} broke the page"
        assert "card turn" in r.text


def test_the_replay_is_reachable_from_the_tables_that_find_a_battle(client):
    """Otherwise the view exists but nothing leads to it: `battles` finds the trace and `scan`
    finds the losing TURN, so both must open the replay — scan's link landing ON that turn."""
    # The battle row IS the link — the whole row navigates (app.js reads data-href) and the id cell
    # is a real <a>, so it works with JavaScript off, tabs to, and opens in a new tab.
    battles = client.get("/partials/battles").text
    assert "/battle?run=" in battles
    assert "data-href=" in battles, "the row is not clickable"
    assert 'class="rowlink"' in battles, (
        "no real <a> in the row — a JS-only row click has no keyboard stop and no open-in-new-tab")

    scan = client.get("/partials/scan", params={"outcome": "loss"}).text
    assert "/battle?run=" in scan
    assert "&start=" in scan and "#inv-" in scan, "scan must open the replay AT the crater"


def test_the_replay_offers_jumps_into_a_long_battle(client):
    """The session's own `notable` block, rendered as links — not re-derived here."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert "worst value drops" in html and "faints" in html
    assert "start=" in html.split("jumps")[1][:600]


def test_the_battles_tab_preselects_the_newest_checkpoint(client, run):
    """A run holds every eval cycle it ever ran. Opening this page essentially always means "what
    is the CURRENT model doing", and an all-steps default answered that with a 200-row cap sliced
    out of an arbitrary mixture of checkpoints."""
    from main.prober.web.app import _newest_step

    summary = ProbeSession(run).run_summary()
    newest = _newest_step(summary)
    assert newest == max(s["step"] for s in summary["steps"])

    html = client.get("/battles").text
    assert f'value="{newest}" selected' in html, "the step filter did not preselect the newest step"
    # ...and the table it first paints is that step's battles, not every step's.
    shown = {r["short_id"] for r in ProbeSession(run).battles(step=newest)}
    other = {r["short_id"] for r in ProbeSession(run).battles() if r["step"] != newest}
    assert shown and other, "fixture needs >1 step for this to mean anything"
    assert any(s in html for s in shown)
    assert not any(o in html for o in other), "an older step's battles are on the first paint"
    # "all steps" must still be one selection away, not gone.
    assert '<option value="">all steps</option>' in html


def test_the_replay_defaults_to_the_newest_checkpoint_not_the_oldest(client, run):
    """`ProbeSession.battles()` is ordered by step ASCENDING, so the naive `rows[0]` default landed
    a visitor on a battle played by the run's OLDEST checkpoint."""
    from main.prober.web.app import _newest_step

    body = client.get("/api/battle-turns").json()
    assert body["step"] == _newest_step(ProbeSession(run).run_summary())


def test_the_turn_dropdown_carries_the_model_free_detail(client):
    """The TUI's per-decision detail, restricted to what needs no checkpoint."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert "<details class=\"more\">" in html
    assert "policy — what else it considered" in html
    assert "class=\"dist\"" in html, "no action distribution"
    assert "raw Showdown protocol" not in html, (
        "the fixture has no *_replay.html sibling, so the protocol panel must not claim one")
    # And it says where the model-dependent analysis actually lives — as a LINK, because /analyze
    # is a web view. It used to point at the Textual TUI, which was retired 2026-08-13.
    assert "/analyze?run=" in html, "the drop-down still dead-ends instead of linking to /analyze"
    assert "main.prober.query analyze" in html, "the CLI equivalent should still be offered"
    assert "the TUI" not in html, "the replay still refers readers to a surface that no longer exists"


def test_the_replay_shows_what_it_expected_the_opponent_to_do(client, run):
    """The v67 α/β read on the card itself, between the board and our choice. That placement is the
    point: it is the only line separating a turn the model played AROUND a threat from one where it
    never saw the move coming, and the board / log / critic numbers are identical in both."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert 'class="expect"' in html
    # Rendered from the session's own numbers, not re-derived here.
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    top = raw["turns"][0]["decisions"][0]["opp_intent"]["alpha"][0]
    assert f'{top["name"]} <span class="p">{top["p"] * 100:.0f}%' in html
    # β is promoted onto the card ONLY on the decision where α actually expects the switch — the
    # fixture's second decision — and the slot is named by the species posterior, not left an index.
    sw = raw["turns"][1]["decisions"][0]["opp_intent"]
    assert sw["top"]["is_switch"] is True
    assert "→ in: slot" in html and "· blissey" in html  # slot index now shown (three hidden slots can share a top-1 belief)
    # The full distribution + β live in the drop-down, next to our own policy distribution.
    assert "opponent intent — what it expected THEM to do" in html
    assert "if they switch, who comes in" in html


def test_a_beta_name_says_whether_it_was_READ_or_BELIEVED(client, run):
    """β points at a SLOT, and where the name came from changes what the row MEANS. The species
    posterior is un-supervised on a slot the board already revealed — measured over a 843-battle
    sweep (2026-08-19) it named a mon not on the opponent's team at all in 73.3% of 6,876 pivots —
    so a posterior-decoded name rendered bare reads as "β predicted this mon" when it is nothing of
    the kind. The fixture holds one of each provenance; both must be distinguishable on the page."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    beta = {c["slot"]: c for t in raw["turns"] for d in t["decisions"]
            if d["opp_intent"] for c in d["opp_intent"]["beta"]}

    # The session resolves the provenance — the page renders it, and derives nothing.
    assert beta[3]["revealed"] is True and beta[3]["caveat"] is None
    assert beta[4]["revealed"] is False and beta[4]["caveat"] == BELIEF_NAME_CAVEAT
    # slot 2 is the one PROMOTED onto the card (α leads with SWITCH there) and it is posterior-named.
    assert beta[2]["revealed"] is False and beta[2]["caveat"] == BELIEF_NAME_CAVEAT

    assert BELIEF_NAME_CAVEAT in html, "a posterior-decoded β name renders with no qualifier"
    assert 'class="tag believed"' in html
    # The board-read name renders PLAIN — caveating the honest half would teach readers to discount
    # it too. Slot 3 is the revealed one; slot 4 sits beside it in the same drop-down list and is not.
    rows = _beta_rows(html)
    assert "believed" not in rows["slot 3"], "a β name read off the BOARD was caveated as a belief"
    assert BELIEF_NAME_CAVEAT in rows["slot 4"], "the posterior-named neighbour lost its caveat"


def test_the_card_says_what_the_OPPONENT_actually_picked(client, run):
    """A prediction is only readable next to its outcome. `α` saying "Drill Peck 41%" means one
    thing when Drill Peck is what came and another when it was not — and until now the difference
    was reachable only by expanding `details` or by reading it back out of the battle log."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    decisions = [d for t in raw["turns"] for d in t["decisions"]]

    # Decision 1: the opponent did something α named — marked in place, and named on its own line.
    hit = decisions[0]["opp_intent"]
    assert hit["actual"] and not hit["actual_unlisted"]
    assert any(o["was_actual"] for o in hit["alpha"]), "the session did not match the actual option"
    assert 'class="opt actual"' in html, "the taken option is not marked in the expect line"
    assert "as expected" in html

    # Decision 2: a move α never listed at all. That is a DIFFERENT failure from ranking it low,
    # and it is the one worth seeing from across the page.
    miss = decisions[1]["opp_intent"]
    assert miss["actual_unlisted"] is True
    assert not any(o["was_actual"] for o in miss["alpha"])
    assert "oppdid miss" in html and "not expected" in html
    # Named the way a human reads it. (The battle log below still prints the recorder's raw id —
    # that is its existing style and not what this line is about.)
    assert "Hydro Pump" in html
    assert 'class="opt">Hydro Pump</span>' in html


def test_a_run_without_the_intent_heads_renders_no_expectation(client, run):
    """Every trace written before v67 carries no `opp_intent` block at all. The card must simply not
    have the line — never an empty one, and never a fabricated 0%."""
    turns = client.get("/api/battle-turns",
                       params={"battle": "step_2000000/aggressive_v2/loss_001"}).json()
    assert all(d["opp_intent"] is None
               for t in turns["turns"] for d in t["decisions"])
    html = client.get("/battle", params={"battle": "step_2000000/aggressive_v2/loss_001"}).text
    assert 'class="expect"' not in html
    assert "opponent intent" not in html


# -- "did it KNOW?" — the awareness layer on the three views that carry it --------------------

def test_the_replay_leads_with_whether_it_saw_the_loss_coming(client, run):
    """The battle-level verdict above the replay, because it changes how the whole game reads: a
    loss the model called 40 turns out is a position it could not convert; one it never saw coming
    is a missed signal. The sentence is the ENGINE's, so this page cannot phrase it its own way."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    aw = raw["awareness"]
    assert aw["knew_by_turn"] is not None, "fixture: this battle must be one it saw coming"
    assert aw["text"] in html, "the page re-worded the engine's verdict instead of printing it"
    assert f"knew @ turn {aw['knew_by_turn']}" in html
    assert "badge blind" not in html


def test_a_blind_loss_is_badged_and_carries_the_stall_signature(client, run):
    """The other shape, and the one that matters: P(loss) never crossed the bar (so the badge says
    BLIND) while catastrophic-band mass piled up under a still-positive mean — exactly what a
    scalar critic cannot show, which is the reason the dist head is read at all."""
    battle = "step_2000000/aggressive_v2/loss_001"
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == battle][0])
    assert raw["awareness"]["blind_loss"] is True
    assert raw["awareness"]["mean_tail_divergence"] >= 0.25
    html = client.get("/battle", params={"battle": battle}).text
    assert "badge blind" in html and "blind loss" in html
    assert "stall signature" in html


def test_every_decision_carries_the_distributions_own_read_under_the_critic_row(client, run):
    """P(loss) per decision, marked from the sustained onset on — so scrolling the replay SHOWS
    where the model started calling it rather than asking the reader to trust the badge."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    rows = [d for t in raw["turns"] for d in t["decisions"]]
    assert 'class="pwin' in html
    assert html.count('class="pwin') == len(rows), "a decision is missing its P(win) strip"
    assert "pwin knew" in html, "the onset marker never rendered"
    assert html.count("pwin knew") == sum(1 for d in rows if d["knew"])
    # Rendered as P(win) — one direction per card — and every value is the session's own p_win.
    for d in rows:
        assert f'aria-label="P(win) {d["p_win"] * 100:.0f}%"' in html


def test_the_replay_shows_the_calibrated_win_prob_beside_v(client, run):
    """P(win) BESIDE V, not instead of it: V is a shaped, discounted return whose zero is not
    'even', so only the calibrated number reads as odds."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert "ΔP" in html and "pp</span>" in html
    # TWO different quantities are both called P(win) on this card — the calibrated HEAD and the
    # distributional strip — so they must stay distinguishable. The strip carries `· dist`.
    assert "P(win) · dist" in html, "the strip must not share a bare name with the head"
    # …and the head is absent entirely on a trace without it, which is the ordinary case. That
    # battle still has a dist strip, so `"P(win)" not in ...` would no longer test anything.
    other = client.get("/battle", params={"battle": "step_2000000/aggressive_v2/loss_001"}).text
    # The marker has to be ROW-SHAPED, and two looser attempts show why: the bare term "ΔP" also
    # appears in the page's legend, and "pp</span>" is a substring of "opp</span>" — the opponent
    # label on every board. `ΔP <span` is the rendered row and nothing else.
    assert "ΔP <span" not in other, "the win-prob head rendered on a trace that has none"
    assert "P(win) · dist" in other, "the distributional strip should still render there"


def test_a_run_with_no_dist_head_renders_no_awareness_rather_than_zeros(client, run):
    """A 0% P(loss) or an un-badged 'not blind' would both be claims the trace cannot support."""
    battle = "step_2000000/aggressive_v2/loss_002"       # the fixture trace with no value_dist
    turns = client.get("/api/battle-turns", params={"battle": battle}).json()
    assert turns["awareness"] is None
    assert all(d["p_loss"] is None for t in turns["turns"] for d in t["decisions"])
    html = client.get("/battle", params={"battle": battle}).text
    assert 'class="ploss' not in html and "awarehead" not in html


def test_scan_rows_say_whether_the_model_saw_each_crater_coming(client, run):
    """The same crater reads completely differently with and without warning, so the verdict sits
    in the row beside the decision that lost it — from the session's own scan output, not a join
    performed here."""
    body = client.get("/partials/scan", params={"outcome": "loss"}).text
    rows = ProbeSession(run).scan(outcome="loss")
    assert any(r["blind_loss"] for r in rows) and any(not r["blind_loss"] for r in rows), (
        "the fixture must carry both shapes for this column to be worth rendering")
    assert "knew @" in body and ">BLIND<" in body
    knew = next(r for r in rows if r["knew_by_turn"] is not None)
    assert f">{knew['knew_by_turn']}</span>" in body


def test_triage_reports_the_awareness_split_beside_the_lever(client, run):
    """Beside, never folded in: the category names WHICH lever to pull, the split says whether the
    model had any warning to act on."""
    body = client.get("/partials/triage").text
    data = ProbeSession(run).triage()
    assert any(c["awareness"]["n_judged"] for c in data["categories"]), "fixture has no verdicts"
    assert ">blind</th>" in body and "median lead" in body
    assert any("REPORTED BESIDE" in c for c in data["caveats"])


def test_the_run_page_carries_the_awareness_panel_against_the_published_baseline(client, run):
    """The run-level readout a generation is judged on — and never a bare number: the gen-10
    baseline ships in the session's own payload so the CLI and this page cannot quote different
    reference points."""
    shell = client.get("/").text
    assert 'hx-get="/partials/awareness' in shell, "the panel must load async, like /scan"
    assert "Did it know?" in shell

    body = client.get("/partials/awareness").text
    agg = ProbeSession(run).awareness_scan()["aggregate"]
    assert "gen-10 baseline" in body
    assert f"{agg['baseline']['blind_loss_fraction'] * 100:.1f}%" in body
    assert f"{agg['blind_loss_fraction'] * 100:.1f}%" in body
    # The small-n and selection caveats are the difference between a reading and a claim.
    assert "direction, not a rate" in body
    assert "biased low" in body and "judge calibration" in body.lower()


def test_the_awareness_panel_says_so_when_the_run_has_no_dist_head(tmp_path):
    """Most runs have none. That is a note about the RUN, not a failure of the probe — and it must
    not render as an empty table or as a row of zeros. Built on its own copy of the fixture rather
    than mutating the shared one, so the state cannot leak into another test."""
    import os as _os

    headless = fixture_run.build(str(tmp_path))
    _os.remove(_os.path.join(headless, "model_config.json"))
    with TestClient(create_app(headless, password="test-only-password")) as c:
        body = c.get("/partials/awareness").text
    assert "no distributional value head" in body
    assert "gen-10 baseline" not in body


def test_every_hand_off_goes_to_the_web_view_not_a_retired_terminal(client, run):
    """`/analyze` IS a web view — it loads the checkpoint and renders faithfulness, beliefs, threat
    tables and saliency in the browser. The replay and the scan table both used to tell readers to
    go and run a CLI command "or the TUI", a surface RETIRED on 2026-08-13, which is how a working
    feature ends up looking absent. Every such hand-off is now a link."""
    replay = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    assert "/analyze?run=" in replay
    assert "the TUI" not in replay

    scan = client.get("/partials/scan", params={"outcome": "loss"}).text
    assert "/analyze?run=" in scan, "a scan row still dead-ends at a command to copy"
    assert "the TUI" not in scan
    # The CLI equivalent stays offered — it is a real second surface, unlike the TUI.
    assert "main.prober.query analyze" in replay and "main.prober.query analyze" in scan


def test_the_critic_row_explains_every_number_it_prints(client):
    """These are the six most cryptic numbers on the page. Each carries a `title`, and because a
    tooltip does not exist on a touch device — and this view is explicitly built for one — the same
    explanations are collected once in a visible legend."""
    html = client.get("/battle", params={"battle": _REPLAY_BATTLE}).text
    for term in ("V(s) — the critic's expected", "VALUE CLIFF", "calibrated", "percentage POINTS",
                 "TD residual", "environment reward"):
        assert term in html, f"the critic row prints a number with no explanation of {term!r}"
    assert "how to read a turn card" in html, "no visible legend — tooltips alone fail on a phone"
    # …and each number is TAPPABLE, because a `title` has no touch equivalent and this view is
    # built to be read on a phone. The class is what app.js's delegated handler keys on.
    assert html.count('class="metric"') >= 6, "the critic row's numbers are not tappable"
    assert 'class="k metric"' in html, "the P(win)·dist label is not tappable"
    # V's zero is the single most misreadable thing on the row, so it is stated in BOTH places.
    assert html.count("not 'even'") + html.count('not "even"') >= 1


def test_the_action_distribution_marks_illegal_actions_without_alarming(client, run):
    """An unavailable action must read as grey/dimmed, never as a red danger value — the same
    distinction the TUI draws with _DISABLED_GREY."""
    turns = client.get("/api/battle-turns", params={"battle": _REPLAY_BATTLE}).json()
    acts = turns["turns"][0]["decisions"][0]["actions"]
    assert acts and any(a["chosen"] for a in acts)
    assert all({"label", "prob", "valid", "chosen"} <= set(a) for a in acts)
    # The session passes the recorder's order through untouched (see the "do NOT re-sort move
    # labels" gotcha); only the template re-orders, and only for display.
    raw = ProbeSession(run).battle_turns(
        [b["id"] for b in ProbeSession(run).battles() if b["short_id"] == _REPLAY_BATTLE][0])
    assert [a["label"] for a in raw["turns"][0]["decisions"][0]["actions"]] == \
           [a["label"] for a in acts]


def test_the_picker_always_contains_the_battle_being_shown():
    """A `<select>` whose options do not contain its value silently displays the FIRST one. With
    the picker capped, arriving from a `scan` deep link to a battle outside the cap would name one
    battle in the dropdown while rendering another below it — a control that lies about what you
    are looking at."""
    from main.prober.web.app import _BATTLE_PICK, _picker_rows

    # Ascending step, exactly as `ProbeSession.battles()` returns them.
    rows = [{"short_id": f"step_{i}/bot/loss_001", "step": i} for i in range(_BATTLE_PICK + 50)]
    oldest = rows[0]
    shown = _picker_rows(rows, oldest)
    assert len(shown) <= _BATTLE_PICK
    assert any(r["short_id"] == oldest["short_id"] for r in shown), (
        "the selected battle is missing from its own picker")
    # And the list itself is newest-first, so the cap drops the OLDEST battles, not the newest.
    newest_first = _picker_rows(rows, rows[-1])
    assert [r["step"] for r in newest_first] == sorted(
        (r["step"] for r in newest_first), reverse=True)
    assert newest_first[0]["step"] == rows[-1]["step"]


def test_the_battle_links_keep_the_selected_run(models_client):
    html = models_client.get("/battle", params={"run": "run_other"}).text
    assert "run=run_other" in html
    assert "ctxstrip" in html and "run_other" in html


def test_the_most_recently_used_run_survives_eviction(tmp_path_factory):
    """LRU, not FIFO: the run you are actually looking at must not be the one thrown away."""
    import shutil
    from main.prober.web.app import _MAX_CACHED_SESSIONS

    root = tmp_path_factory.mktemp("lru")
    staging = tmp_path_factory.mktemp("lrustage")
    names = []
    for i in range(_MAX_CACHED_SESSIONS + 1):
        built = fixture_run.build(str(staging / f"s{i}"))
        dest = root / f"run_{i}"
        shutil.move(built, str(dest))
        names.append(dest.name)

    app = create_app(str(root))
    with TestClient(app) as c:
        for name in names[:_MAX_CACHED_SESSIONS]:
            c.get("/api/run", params={"run": name})
        keep = names[0]
        c.get("/api/run", params={"run": keep})       # touch the oldest -> now newest
        c.get("/api/run", params={"run": names[-1]})  # force one eviction

        cached = {p.rsplit("/", 1)[-1] for p in app.state.sessions}
        assert keep in cached, "the just-used run was evicted — that is FIFO, not LRU"


# -- /analyze: the one model-loading view ------------------------------------------------------
#
# Two things make this view different from every other page here, and both are tested rather than
# assumed. (1) It LOADS A CHECKPOINT, so its normal outcome on an archived run is `ArchDriftError`
# — measured 79/79 runs — and that error is a multi-line DIAGNOSIS ending in the exact
# `git checkout` to re-probe from, which the page has to render WHOLE. (2) Everything it shows is
# flag-gated: most panels are `None` unless the run trained the head, so "absent" must mean "that
# head was off", never a silently swallowed failure.
#
# The populated cases run against a monkeypatched `session.analyze`, exactly as the job tests
# replace `falsify_scan`: the fixture run has no loadable checkpoint (by design — see
# `fixture_run.py`), and re-running a real policy is `engine_test.py`'s job, not this file's.

_ANALYZE_BATTLE = "step_4000000/heuristic2/loss_003"

_TIMELINE = [
    {"side": "we", "kind": "move", "order_certain": True,
     "text": "we thunderbolt did 31% (tyranitar 100% → 69%)"},
    {"side": "opp", "kind": "move", "order_certain": True,
     "text": "opp earthquake did 22% (zapdos 100% → 78%) ⚡CRIT"},
]

_FULL_ANALYSIS = {
    "meta": {"step": 4000000, "battle_id": "b1", "result": "LOSS", "turns": 12,
             "n_invocations": 3, "summary_path": "/x/s.json", "npz_path": "/x/s.npz"},
    "inv_index": 1, "turn": 7, "phase": "move_selection",
    "our_species": "zapdos", "opp_species": "tyranitar", "chosen": "thunderbolt",
    "has_state": True,
    # DELIBERATELY not in probability order: the recorded order IS the action order and the table
    # must pass it through (see the "do NOT re-sort move labels" gotcha in prober/CLAUDE.md).
    "actions": [
        {"label": "switch:blissey", "valid": True, "recorded": 0.10, "rerun": 0.14,
         "is_chosen": False},
        {"label": "thunderbolt", "valid": True, "recorded": 0.70, "rerun": 0.55, "is_chosen": True},
        {"label": "hiddenpower", "valid": False, "recorded": 0.20, "rerun": 0.31,
         "is_chosen": False},
    ],
    "matchups": {"multipliers": [2.0, 1.0, 2.0, 4.0],
                 "move_labels": ["thunderbolt", "hiddenpower", "spikes", "roar"],
                 "applicable": [True, True, False, False]},
    "sweep": {"chosen_label": "thunderbolt", "request_slot": 0, "baseline_p_switches": 0.22,
              "rows": [{"multiplier": 0.0, "p_chosen": 0.1, "p_switches": 0.4},
                       {"multiplier": 4.0, "p_chosen": 0.9, "p_switches": 0.05}]},
    "saliency": {"overall_mean_abs": 0.01,
                 "blocks": [{"name": "our_team(696)", "mean_abs": 0.03, "total_abs": 20.9},
                            {"name": "turn_history(1113)", "mean_abs": 0.004, "total_abs": 4.4}]},
    "value_saliency": {"overall_mean_abs": 0.02,
                       "blocks": [{"name": "our_team(696)", "mean_abs": 0.05, "total_abs": 34.8}]},
    "threats": {"present": True, "revealed_frac": 0.33, "max_incoming": 4.0,
                "per_our_slot_max": [4.0, 1.0, 0.5, 0.0, 0.0, 0.0]},
    "incoming": {"present": True, "max_pko": 0.81, "active_pko": 0.62, "active_exp": 0.44,
                 "active_outspeed": 0.9, "per_slot_pko": [0.62, 0.1, 0.0, 0.0, 0.0, 0.0],
                 "recovery_rate": 0.2, "cures_status": 0.05, "recovery_known": 1.0,
                 "active_pko_nocrit": 0.5, "threat_revealed": 1.0},
    "warnings": [],
    "outcome": {"our": {"action": "thunderbolt", "hp_delta": "-22%"},
                "opp": {"action": "earthquake", "hp_delta": "-31%"},
                "reward": {"total": -1.5, "hp": -1.0, "faint": -0.5},
                "events": ["our:zapdos:fainted"], "timeline": _TIMELINE},
    "value": {"recorded": 3.5, "rerun": 3.2, "next_recorded": -4.0, "delta": -7.5,
              "popart_mu": -3.6, "popart_sigma": 4.0, "normalized_recorded": 1.77,
              "normalized_rerun": 1.7, "td_residual": -9.0,
              "td_phrase": "much worse than the critic expected"},
    "win_prob": {"recorded": 0.61, "next_recorded": 0.2, "delta": -0.41},
    "value_dist": {"probs": [0.1, 0.4, 0.5], "support": [-10.0, 0.0, 10.0], "mean": 2.0,
                   "std": 3.0, "p10": -10.0, "p50": 0.0, "p90": 10.0, "entropy": 0.94,
                   "bimodality": 0.35, "mean_real": 4.4},
    "rerun_argmax": "switch:blissey", "agrees": False,
    "flags": ["switch", "faint"], "cure_options": ["refresh"],
    "board": {"ours": {"active_species": "zapdos", "active_hp": "78%", "status": "PAR",
                       "boosts": "spa:+1", "moves": ["thunderbolt"], "bench": [],
                       "item": "leftovers"},
              "opp": {"active_species": "tyranitar", "active_hp": "69%", "status": "",
                      "boosts": "", "moves": [], "bench": [], "item": ""}},
    "next_board": None,
    "obs_mismatch": [2667, 2669],
    "field": {"weather": "sandstorm", "spikes_opp": 1, "wish_our": True},
    "belief": None,
    # The species-clause reading, INCOHERENT on purpose: slots 2 and 4 both name blissey (which no
    # gen3 team allows) and slot 4 additionally carries mass on the already-revealed tyranitar.
    "exclusive_belief": {
        "slots": [
            {"slot": 2, "top": [["blissey", 0.55], ["snorlax", 0.25]],
             "raw_top1": "blissey", "raw_top1_prob": 0.62,
             "adj_top1": "blissey", "adj_top1_prob": 0.55,
             "differs": False, "total_variation": 0.07,
             "hypothesis": "blissey", "hypothesis_differs": False},
            {"slot": 4, "top": [["snorlax", 0.41], ["blissey", 0.30]],
             "raw_top1": "tyranitar", "raw_top1_prob": 0.44,
             "adj_top1": "snorlax", "adj_top1_prob": 0.41,
             "differs": True, "total_variation": 0.46,
             "hypothesis": "snorlax", "hypothesis_differs": True},
        ],
        "team_hypothesis": ["blissey", "snorlax"],
        "revealed": ["tyranitar"],
        "max_expected_count": 1.24, "illegal_mass": 0.24, "duplicate_top1": 1,
        "revealed_leak_max": 0.44, "converged": True, "iterations": 31, "coherent": False,
    },
    "opp_intent": {"alpha": [{"name": "earthquake", "p": 0.5, "is_switch": False},
                             {"name": "SWITCH", "p": 0.3, "is_switch": True}],
                   # ONE OF EACH β naming provenance, because the three render differently: a
                   # board-read name (plain), a posterior decode (carries the engine's caveat), and
                   # a slot no species head could name at all (a bare index, already honest).
                   "beta": [{"slot": 2, "p": 0.6, "species": "blissey", "revealed": True,
                             "caveat": None},
                            {"slot": 3, "p": 0.3, "species": "porygon2", "revealed": False,
                             "caveat": BELIEF_NAME_CAVEAT},
                            {"slot": 4, "p": 0.2, "species": None, "revealed": False,
                             "caveat": None}],
                   "top": {"name": "earthquake", "p": 0.5, "is_switch": False},
                   "switch_p": 0.3, "text": "it expected earthquake (50%)"},
    "belief_truth": {"mons": [
        {"species": "tyranitar", "revealed": True, "guess": [], "guessed_right": False,
         "true_rank": -1},
        {"species": "blissey", "revealed": False, "guess": [["blissey", 0.5], ["snorlax", 0.2]],
         "guessed_right": True, "true_rank": 1},
        {"species": "skarmory", "revealed": False,
         "guess": [["forretress", 0.4], ["skarmory", 0.3]], "guessed_right": False,
         "true_rank": 2},
        {"species": "gengar", "revealed": False, "guess": [["misdreavus", 0.4]],
         "guessed_right": False, "true_rank": -1}], "n_hidden": 3, "n_correct": 1},
    "opp_full_team": None,
    "damage_op": {
        "incoming": [{"phys": {"low": 0.3, "high": 0.4, "crit": 0.06, "pko": 0.2, "acc": 1.0},
                      "spec": {"low": 0.1, "high": 0.2, "crit": 0.06, "pko": 0.0, "acc": 1.0},
                      "p_outspeed": 0.9, "provenance": 1.0}] * 6,
        "choice_band": {"phys_high_cb": [0.6] * 6, "phys_pko_cb": [0.4] * 6, "p_cb": 0.18},
        "outgoing": {"moves": [{"low": 0.5, "high": 0.6, "crit": 0.07, "pko": 0.55}] * 4,
                     "p_outspeed": 0.9, "secondary": [{"par": 0.1}] * 4},
        "status_landing": [{"p_land": 0.75, "known": 1.0}] * 4,
        "outgoing_matrix": None,
        "incoming_matrix": {
            "moves": [{"move": "earthquake", "belief": 0.9, "accuracy": 1.0, "is_phys": 1.0,
                       "effect": [], "secondary": []},
                      {"move": "icebeam", "belief": 0.6, "accuracy": 0.9, "is_phys": 0.0,
                       "effect": [], "secondary": []}],
            # per OUR mon (6) × per candidate move (2). Slot 0 is immune to earthquake — the cell
            # must read "safe", not a damage range.
            "per_defender": [[{"low": 0.0, "high": 0.0, "crit": 0.0, "pko": 0.0, "type_mult": 0.0,
                               "status_lands": 0.0},
                              {"low": 0.2, "high": 0.3, "crit": 0.1, "pko": 0.05,
                               "type_mult": 1.0, "status_lands": 0.0}]] * 6},
        "outgoing_matrix_all": None},
    "move_belief": {"opp": [{"slot": 0, "species": "tyranitar",
                             "revealed": [["earthquake", 0.99]],
                             "believed": [["rockslide", 0.55]]}],
                    "our_labels": [[0, "zapdos", True], [1, "blissey", False],
                                   [2, "skarmory", False], [3, "gengar", False],
                                   [4, "swampert", False], [5, "jynx", False]]},
    "spread_belief": {"slots": [{"slot": 0, "species": "tyranitar",
                                 "rows": [{"stat": "atk", "believed": 385.0, "true": 305.0,
                                           "prior": 320.0}],
                                 "nature": "adamant", "ev_note": "atk252/hp4", "matched": True}],
                      "n_slots": 1, "mean_abs_err": 40.5},
    "refine_trajectory": None,
    "switch_in_outgoing": {"opp_species": "tyranitar", "opp_hp": "69%",
                           "rows": [{"species": "swampert", "hp": "100%", "move": "earthquake",
                                     "low": 42.0, "high": 50.0, "pko": 0.6, "type_mult": 2.0,
                                     "outspeed": 0.2}]},
    "opp_switched_to": "skarmory",
    "model_resolution": {"path": "/x/ck.zip", "tier": "nearest", "detail": "checkpoint 3.2M",
                         "manifest": {"git_hash": "deadbeef", "arch_signature": "gen3_fixture_v1"},
                         "dropped_kwargs": ["spread_belief_nature_marginalize"]},
    "protocol": ["|move|p1a: Zapdos|Thunderbolt|p2a: Tyranitar",
                 "|-damage|p2a: Tyranitar|69/100"],
}

# The same decision with every optional head OFF — which is what most runs actually look like.
_BARE_ANALYSIS = dict(
    _FULL_ANALYSIS,
    matchups=None, sweep=None, saliency=None, value_saliency=None, threats=None, incoming=None,
    value_dist=None, win_prob=None, opp_intent=None, belief=None, belief_truth=None,
    damage_op=None, move_belief=None, spread_belief=None, switch_in_outgoing=None,
    obs_mismatch=None, opp_switched_to=None, cure_options=[], protocol=[],
    model_resolution={"path": None, "tier": "exact", "detail": "eval snapshot",
                      "manifest": None, "dropped_kwargs": []},
)


def _stub_analyze(client, payload):
    sess = _the_session(client)
    sess.analyze = lambda *a, **kw: payload
    return sess


def _fragment(client, **params):
    params.setdefault("battle", _ANALYZE_BATTLE)
    params.setdefault("inv", "1")
    r = client.get("/partials/analyze", params=params)
    assert r.status_code == 200, r.text[:400]
    return r.text


@pytest.mark.parametrize("path", ["/battle", "/analyze"])
def test_a_run_with_no_traces_is_an_empty_state_not_a_404(tmp_path, path):
    """The FIRST thing a fresh run shows, so it must not look broken.

    Both battle-addressed pages resolve a battle before rendering, and a run that has captured
    nothing yet has none — which used to surface as a 404, i.e. indistinguishable from a bad link on
    a perfectly healthy run. And it is not an edge case: the app opens the NEWEST run by default,
    and a newly-launched run has no traces until its first eval cycle (gen-9 sat in exactly this
    state for hours). A real 404 is still a real 404 — that is the next test."""
    run = os.path.join(str(tmp_path), "run_empty")
    os.makedirs(os.path.join(run, "eval_traces"))
    with open(os.path.join(run, "metadata.json"), "w") as fh:
        json.dump({"gamma": 0.99}, fh)

    app = create_app(run)
    with TestClient(app) as c:
        r = c.get(path)
        assert r.status_code == 200, f"{path} on a traceless run: {r.status_code}"
        assert "captured no battle traces yet" in r.text
        assert "run summary" in r.text, "it must point somewhere that DOES have something to show"
        assert "Traceback" not in r.text


def test_a_battle_that_does_not_exist_is_still_a_404(client):
    """The empty-state branch must not have swallowed the real one: an unknown battle token is a
    genuine 404, and the message never echoes the token back (a rendered error must not become an
    oracle for what exists)."""
    r = client.get("/analyze", params={"battle": "step_9/nope/loss_999"})
    assert r.status_code == 404
    assert "nope" not in r.text and "loss_999" not in r.text


def test_the_analyze_page_renders_and_defers_the_work_to_a_fragment(client):
    """Unlike `/battles` and `/triage` this one may NOT arrive populated: it deserializes a
    checkpoint. Same answer as `/scan` — arrive, then fill in, and say what is being waited on."""
    html = client.get("/analyze").text
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "Decision analysis" in html
    assert 'hx-get="/partials/analyze' in html and 'hx-trigger="load"' in html
    assert "Loading the checkpoint" in html, "the waiting state must explain the wait"
    assert "faithfulness" not in html, "the analysis itself must not be on the first paint"


def test_the_analyze_page_is_a_linkable_get_form(client):
    """"Look at this decision" is a thing you send someone, so the battle and the inv are plain GET
    params — no HTMX on the selection, and the URL survives a reload."""
    html = client.get("/analyze", params={"battle": _ANALYZE_BATTLE, "inv": "1"}).text
    assert 'action="/analyze"' in html and 'method="get"' in html
    assert f'value="{_ANALYZE_BATTLE}" selected' in html
    assert 'name="inv"' in html and 'value="1"' in html


def test_a_bad_inv_value_does_not_break_the_analyze_page(client):
    """A hand-edited URL is a browser control by another name."""
    for inv in ("", "abc", "-3"):
        r = client.get("/analyze", params={"battle": _ANALYZE_BATTLE, "inv": inv})
        assert r.status_code == 200, f"inv={inv!r} broke the page"


def test_api_analyze_returns_the_session_result_unreshaped(client):
    """The package's one rule at this endpoint: the handler is a pass-through, not a renderer."""
    _stub_analyze(client, _FULL_ANALYSIS)
    got = client.get("/api/analyze", params={"battle": _ANALYZE_BATTLE, "inv": 1}).json()
    assert got == _FULL_ANALYSIS


@pytest.mark.parametrize("attack", [
    "../../../etc/passwd",
    "/etc/passwd",
    "%2e%2e%2fetc%2fpasswd",
    "step_4000000/heuristic2/../../../loss_003",
    "step_1/nope/loss_999",
])
def test_an_analyze_battle_token_is_never_joined_to_a_path(client, attack):
    """`runs.py`'s membership rule, repeated one level down — the same reason `/battle` needs it:
    `ProbeSession._battle` falls back to `build_trace_tree(battle_id)` for an unrecognised id and
    will happily open a `*_summary.json` belonging to another run."""
    api = client.get("/api/analyze", params={"battle": attack, "inv": 0})
    assert api.status_code == 404
    assert "passwd" not in api.text and "etc" not in api.json().get("error", "")
    for path in ("/analyze", "/partials/analyze"):
        assert client.get(path, params={"battle": attack}).status_code == 404, path


def test_the_real_analyze_on_this_fixture_renders_a_diagnosis_not_a_500(client):
    """The fixture has no loadable checkpoint (by design), and neither does any archived run:
    measured 2026-08-13, 79 of 79 fail to load under current code. So a failed model load is an
    ordinary state of the data and must READ as one."""
    body = _fragment(client, inv="0")
    assert 'data-analysis="error"' in body
    assert "cannot be re-run under the current code" in body
    # ...and it points at the views that DO work on this run rather than dead-ending.
    assert "/scan?run=" in body and "/triage?run=" in body


def test_the_arch_drift_message_renders_whole_including_its_git_checkout_line(client):
    """The message is a multi-line diagnosis written for a human whose LAST useful line is the
    exact commit to re-probe from. Collapsing it to "analysis failed" throws away the only part
    that says what to do next — so this pins the whole thing through, newlines included."""
    from main.prober.model import ArchDriftError

    message = ("This checkpoint cannot be re-run under the current code:\n"
               "  /models/run_x/checkpoints/checkpoint_1_steps.zip\n"
               "\n"
               "  obs dim        trained 2667  ·  code builds 2669\n"
               "  · probe from the commit it was trained on:  git checkout abc123def\n"
               "\n"
               "MODEL-FREE views need none of this and work on every archived run:\n"
               "  scan · triage · turns · overview · find · falsify · calibration")

    def boom(*a, **kw):
        raise ArchDriftError(message, saved_obs_dim=2667, current_obs_dim=2669,
                             git_hash="abc123def")

    _the_session(client).analyze = boom
    body = _fragment(client)
    assert "git checkout abc123def" in body, "the one actionable line was dropped"
    assert "obs dim        trained 2667" in body
    # The line breaks have to SURVIVE — `.err` is `white-space: pre-wrap` and the message is
    # rendered as one text node, so a browser lays it out as written.
    assert 'class="err"' in body
    assert body.count("\n", body.index("cannot be re-run"), body.index("git checkout")) >= 4


def test_analyze_panels_self_hide_when_their_head_was_off(client):
    """Most of this view is flag-gated (`--damage-op`, `--move-belief-mode`, `--spread-belief`,
    `--value-dist-mode`, `--win-prob-mode`, `--opp-intent-coef`). An absent panel must mean "that
    head was off", never an empty box that reads as a broken probe."""
    _stub_analyze(client, _BARE_ANALYSIS)
    bare = _fragment(client)
    for gone in ("beliefs", "threats", "saliency", "intervention", "P(win)",
                 "predicted return distribution", "OBS MISMATCH", "DROPPED FLAGS",
                 "opp pivoted", "raw Showdown protocol"):
        assert gone not in bare, f"{gone!r} rendered on a run whose head was off"
    # ...while the always-present parts still do.
    assert "faithfulness" in bare and "what happened" in bare and "the critic" in bare

    _stub_analyze(client, _FULL_ANALYSIS)
    full = _fragment(client)
    for shown in ("beliefs", "threats", "saliency", "intervention", "P(win)",
                  "predicted return distribution"):
        assert shown in full, f"{shown!r} missing on a run that trained it"


def _beta_rows(html: str) -> dict:
    """`{"slot N" -> that ROW's html}` for the β candidate list — isolated to the list under the β
    heading first, because "believed" is a word this page uses elsewhere (the belief panel's
    "believed unseen" heading), and a whole-page substring test would read that as a β caveat."""
    start = html.index("if they switch, who comes in")
    section = html[start:html.index("</ul>", start)]
    rows = {}
    for chunk in section.split("<li>")[1:]:
        row = chunk.split("</li>")[0]
        for n in ("slot 2", "slot 3", "slot 4"):
            if n in row:
                rows[n] = row
    return rows


def test_analyze_marks_a_BELIEVED_beta_name_and_leaves_a_REVEALED_one_plain(client):
    """/analyze renders β's FULL candidate list, so it is where the provenance mix is visible at
    once. `_FULL_ANALYSIS` carries one of each: a board-read name (plain), a posterior decode
    (caveated), and a slot with no species head at all (a bare index, nothing to qualify).

    The caveat STRING is the engine's (`BELIEF_NAME_CAVEAT`) — a sentence one surface learns must
    not go missing on another, and a second surface authoring its own wording is how two surfaces
    end up saying different things about the same row."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    rows = _beta_rows(body)

    # The posterior decode — porygon2 is precisely the name that was read as a β PREDICTION in the
    # analysis this fix came out of, on a turn whose slot held a revealed Salamence.
    assert "porygon2" in rows["slot 3"] and BELIEF_NAME_CAVEAT in rows["slot 3"]
    assert 'class="tag believed"' in rows["slot 3"]
    # The board-read name is a fact and stays plain.
    assert "blissey" in rows["slot 2"] and "believed" not in rows["slot 2"]
    # A slot no species head could name renders as a bare index — a caveat needs a name to qualify.
    assert "believed" not in rows["slot 4"]


def test_the_timeline_prints_the_engines_own_sentence_verbatim(client):
    """`engine.timeline_entry_text` exists so no second surface re-derives the battle-log line from
    the structured fields — the drift the engine/renderer split is there to prevent."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    for entry in _TIMELINE:
        assert entry["text"] in body, entry["text"]


def test_an_unrecorded_move_order_is_never_rendered_as_a_sequence(client):
    """`order_certain=False` means both sides moved and the recorder never captured who went
    first, so a numbered list would be a quiet lie about the sequence."""
    _stub_analyze(client, _FULL_ANALYSIS)
    ordered = _fragment(client)
    assert '<ol class="log">' in ordered
    assert "move order not recorded" not in ordered

    unordered_timeline = [dict(e, order_certain=False) for e in _TIMELINE]
    payload = dict(_FULL_ANALYSIS,
                   outcome=dict(_FULL_ANALYSIS["outcome"], timeline=unordered_timeline))
    _stub_analyze(client, payload)
    body = _fragment(client)
    assert '<ol class="log">' not in body, "an unordered log must not be a numbered list"
    assert "log unordered" in body
    assert "(move order not recorded)" in body
    for entry in unordered_timeline:
        assert entry["text"] in body


def test_the_faithfulness_table_keeps_the_recorded_action_order(client):
    """The recorded `actions` are ALREADY in action-index order (the recorder keys move slot m on
    `legal.move_ids[m]`). A re-sort onto the per-mon moveset order once transposed the labels and
    produced a spurious `disagree` — see the gotcha in src/main/prober/CLAUDE.md."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    positions = [body.index(r["label"] + "</td>") for r in _FULL_ANALYSIS["actions"]]
    assert positions == sorted(positions), "the action rows were re-ordered for display"
    # The signed delta is what makes recorded-vs-rerun readable at a glance.
    assert "-0.150" in body, "no signed re-run − recorded delta"


def test_a_phantom_type_multiplier_renders_as_a_dash(client):
    """The obs computes a multiplier for EVERY request slot, including non-damaging ones, where it
    is an artefact — a confident "2.00×" on Spikes is a misreading waiting to happen."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    # spikes/roar are `applicable=False` and carry 2.0 / 4.0 in the same array.
    spikes = body.index("spikes")
    assert "—" in body[spikes:spikes + 300]
    assert "2.00×" not in body[spikes:spikes + 300]


def test_the_intervention_says_so_when_the_chosen_action_is_not_a_move(client):
    """`InterventionSweep.applicable` is a @property, so `asdict` DROPS it — the test has to be
    `request_slot >= 0`, not a missing key that would silently read as False forever."""
    _stub_analyze(client, dict(_FULL_ANALYSIS,
                               sweep=dict(_FULL_ANALYSIS["sweep"], request_slot=-1, rows=[])))
    body = _fragment(client)
    assert "not a move" in body
    assert "0.0×" not in body


def test_the_switch_in_percentages_are_not_scaled_twice(client):
    """`switch_in_outgoing` rows are ALREADY percentages (0–100) while the operator's own
    low/high are FRACTIONS (0–1). A uniform ×100 across the panel turns 42% into 4200%."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "42–50%" in body
    assert "4200" not in body


def test_the_cpu_decodes_dim_only_when_the_operator_subsumes_them(client):
    """The graceful-degradation contract: the observation decodes go secondary when the
    DamageOperator is present and back to full strength when there is no operator."""
    _stub_analyze(client, _FULL_ANALYSIS)
    assert 'class="cpudecodes dim"' in _fragment(client)

    _stub_analyze(client, dict(_FULL_ANALYSIS, damage_op=None, switch_in_outgoing=None))
    no_op = _fragment(client)
    assert "cpudecodes" in no_op
    assert "dim" not in no_op.split('class="cpudecodes')[1][:40]


def test_the_incoming_matrix_is_a_real_heatmap_and_marks_immunity(client):
    """The biggest win over the terminal: which opponent move threatens which of our mons, as a
    grid tinted by P(KO). A type-immune cell must read `safe`, not a damage range."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert 'class="heat"' in body
    assert "earthquake" in body and "icebeam" in body
    # Our mons label the columns from the move-belief's own our_labels, active marked.
    assert "▶ zapdos" in body and "swampert" in body
    assert ">safe<" in body, "a type-immune cell rendered a damage number"
    assert "rgba(208, 128, 106," in body, "the P(KO) tint is what makes it a heatmap"


def test_the_td_residual_never_appears_without_its_plain_language_gloss(client):
    """The rule this view inherits from the TUI: the ML term is always paired with the engine's
    own sentence, so the number is self-explaining."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "much worse than the critic expected" in body
    assert "TD δ" in body


def test_the_three_unreliability_banners_render(client):
    """Each of these means "stop before quoting a number below", so none of them may be a
    footnote: an obs-dim mismatch, a dropped extractor flag, and an opponent pivot that makes the
    damage tables be about the wrong defender."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "OBS MISMATCH" in body and "2667" in body and "2669" in body
    assert "DROPPED FLAGS" in body and "spread_belief_nature_marginalize" in body
    assert "opp pivoted tyranitar → skarmory" in body
    assert "RESOLVED against <strong>skarmory" in body


def test_the_belief_markers_keep_their_three_way_meaning(client):
    """✓ top-1 right · ≈ the true mon is in the belief but not top-1 · ✗ not in the top-k at all.
    Collapsing the middle case into "wrong" loses the near-miss, which is the whole signal in a
    belief that is sharpening."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "marker hit" in body and "marker near" in body and "marker miss" in body
    assert "1/3 hidden mons" in body
    assert "(#2)" in body, "the true species' rank is the near-miss's magnitude"


def test_the_gpu_cpu_provenance_distinction_survives_from_the_tui(client):
    """A signal the model computed for itself and a decode the prober did from the observation
    answer different questions; a panel that mixes them without saying so reads as one."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "🔷 GPU" in body and "📋 CPU" in body


def test_analyze_is_in_the_nav_and_the_where_to_start_card(client):
    """We are retiring the TUI, so this view has to be reachable without knowing it exists."""
    from main.prober.web.app import _NAV
    assert ("/analyze", "analyze") in _NAV
    html = client.get("/").text
    assert "/analyze" in html
    assert "Why did it choose that" in html


def test_the_analyze_view_carries_the_run_into_its_links(models_client):
    html = models_client.get("/analyze", params={"run": "run_other"}).text
    assert "run=run_other" in html
    assert "ctxstrip" in html and "run_other" in html


# -- /analyze § counterfactual: lookahead · better-line · replay-to-end ------------------------
#
# The three probes that re-PLAY the decision rather than read it. Everything here is exercised with
# the session method REPLACED, exactly as the falsify/calibration job tests are: the re-roll
# machinery needs Node and belongs to `falsifier_integration_test.py` / `better_line_integration_test.py`.
# What these pin is what this package actually owns — the gate, the job lifecycle, the FIELDS
# reaching the session, and whether the rendered page carries each number's MEANING and its caveat.

_CF_PATHS = ["/partials/job/lookahead", "/partials/job/better-line",
             "/partials/job/replay-counterfactual"]
_CF_API = ["/api/jobs/lookahead", "/api/jobs/better-line", "/api/jobs/replay-counterfactual"]

_FAKE_LOOKAHEAD = {
    "inv": 1, "turn": 7, "side": "p1",
    "chosen": {"action": 7, "label": "thunderbolt", "choice": "move 2"},
    "recorded_value": 3.5, "recorded_next_value": -4.0, "baseline_value": -4.0, "n_seeds": 0,
    # Already ranked best-first by the session — the renderer must NOT re-sort.
    "candidates": [
        {"action": 1, "label": "switch:swampert", "choice": "switch 2", "is_chosen": False,
         "value_crn": 1.25, "value_mean": None, "value_std": None, "n_evaluated": 1,
         "terminal_frac": 0.0, "terminal": None, "win_prob_crn": 0.55, "value_dist_crn": None,
         "delta_v": 5.25},
        {"action": 7, "label": "thunderbolt", "choice": "move 2", "is_chosen": True,
         "value_crn": -4.0, "value_mean": None, "value_std": None, "n_evaluated": 1,
         "terminal_frac": 0.0, "terminal": None, "win_prob_crn": 0.2, "value_dist_crn": None,
         "delta_v": 0.0},
        {"action": 8, "label": "explosion", "choice": "move 3", "is_chosen": False,
         "value_crn": None, "value_mean": None, "value_std": None, "n_evaluated": 0,
         "terminal_frac": 1.0, "terminal": "loss", "win_prob_crn": None, "value_dist_crn": None,
         "delta_v": None},
    ],
    "best_alternative": "switch:swampert", "best_delta_v": 5.25,
}

_FAKE_BETTER_LINE = {
    "inv": 1, "turn": 7, "side": "p1", "depth": 2, "beam": 3, "top_k": 4,
    "opp_model_used": "reloaded:self_model_approx (+1 interior plies → sim default)",
    "interior_opponent": "reloaded:self_model_approx",
    "chosen": {"action": 7, "label": "thunderbolt", "choice": "move 2"},
    "recorded_value": 3.5, "recorded_next_value": -4.0, "baseline_value": -4.0,
    "candidates": [
        {"action": 1, "label": "switch:swampert", "choice": "switch 2", "is_chosen": False,
         "value": 1.25, "backup": 2.75, "terminal": None, "delta_v": 5.25, "win_prob": 0.61,
         "principal_variation": [{"depth": 1, "action": 1, "value": 1.25, "terminal": None},
                                 {"depth": 2, "action": 6, "value": 2.75, "terminal": None}]},
        {"action": 7, "label": "thunderbolt", "choice": "move 2", "is_chosen": True,
         "value": -4.0, "backup": -3.1, "terminal": None, "delta_v": 0.0, "win_prob": 0.2,
         "principal_variation": [{"depth": 1, "action": 7, "value": -4.0, "terminal": None}]},
    ],
    "best_alternative": {
        "action": 1, "label": "switch:swampert", "choice": "switch 2", "backup": 2.75,
        "terminal": None, "delta_v": 5.25, "win_prob": 0.61,
        "principal_variation": [{"depth": 1, "action": 1, "value": 1.25, "terminal": None},
                                {"depth": 2, "action": 6, "value": 2.75, "terminal": None}]},
}

_FAKE_REPLAY_ONE = {
    "inv": 1, "turn": 7, "trainee_side": "p1", "recorded_result": "loss",
    "chosen": {"action": 7, "label": "thunderbolt", "choice": "move 2"},
    "substitute": {"action": 1, "label": "switch:swampert", "choice": "switch 2"},
    "opponent_source": "self_model_approx", "n_rollouts": 1, "wins": 1, "losses": 0,
    "win_rate": 1.0, "win_rate_ci": [0.2065, 1.0], "outcomes": {"win": 1},
    "deterministic_line": True,
    "winning_trajectory": [{"turn": 7, "events": ["we switch to swampert",
                                                  "opp earthquake did 12%"]},
                           {"turn": 8, "events": ["we icebeam — opp salamence fainted"]}],
    "losing_trajectory": None,
    "caveats": [
        "The opponent plays its policy FRESH past the divergence; the recorded opponent's logged "
        "choices are invalid once our move changes.",
        "Single realized-dice line (n_rollouts=1) — NOT a probability. Raise --rollouts for a "
        "Monte-Carlo win-rate ± CI over resampled post-divergence dice.",
    ],
}


def _cf_forms(client):
    """The three launch forms, rendered on a POPULATED analyze fragment. They live in the branch
    that has an analysis — on an arch-drift run there is no model to re-roll with either."""
    _stub_analyze(client, _FULL_ANALYSIS)
    return _fragment(client)


def test_the_counterfactual_probes_are_launchable_from_the_decision_they_are_about(client):
    """They are PER-DECISION, so they belong where the battle, the inv and the legal actions
    already are — not on a page of their own that would ask for all three again."""
    body = _cf_forms(client)
    for path in _CF_PATHS:
        assert f'hx-post="{path}"' in body, path
    # ...pre-filled with THIS decision, and pointed at the box the job poller targets.
    assert f'name="battle" value="{_ANALYZE_BATTLE}"' in body
    assert 'name="inv" value="1"' in body
    assert 'hx-target="#job"' in body and 'id="job"' in body
    # Nothing starts on its own: each of these spends real CPU.
    assert "nothing run yet" in body


def test_the_replay_action_picker_is_the_decisions_own_action_list(client):
    """`replay_counterfactual` needs an action INDEX, and the only honest source for it is the
    recorded action list this page already renders — in action-index order, illegal ones disabled
    rather than hidden, and with no default, because "replay something else" has to say what."""
    body = _cf_forms(client)
    assert 'name="action" required' in body
    assert "— pick an action —" in body
    for i, act in enumerate(_FULL_ANALYSIS["actions"]):
        assert f'value="{i}"' in body
        assert act["label"] in body
    # The third fixture action is illegal; it is offered as context but cannot be picked.
    assert "disabled>" in body and "— illegal" in body
    assert "(played)" in body, "the action that WAS played has to be distinguishable"


@pytest.mark.parametrize("path", _CF_PATHS)
def test_a_counterfactual_submit_is_locked_without_the_password(client, path):
    """Reading is anonymous; spending CPU is not. Each of these spawns Node for seconds to
    minutes beside a live trainer, so the rule that gates falsify/calibration gates them too."""
    r = client.post(path, data={"battle": _ANALYZE_BATTLE, "inv": "1", "action": "1"})
    assert r.status_code == 200, "a locked visitor gets a way in, not an error"
    assert 'data-job-status="locked"' in r.text
    assert "shared password" in r.text
    assert "/login?next=" in r.text


@pytest.mark.parametrize("path", _CF_API)
def test_the_counterfactual_job_api_is_403_without_the_password(client, path):
    r = client.post(path, params={"battle": _ANALYZE_BATTLE, "inv": 1, "action": 1})
    assert r.status_code == 403
    assert "password" in r.json()["error"]


def _run_cf(client, path, method, payload, **fields):
    """Submit one counterfactual from the page, wait for it, and return (rendered html, kwargs the
    session was called with)."""
    _unlock(client)
    seen = {}

    def stub(*a, **kw):
        seen["args"], seen["kwargs"] = a, kw
        return payload

    setattr(_the_session(client), method, stub)
    data = {"battle": _ANALYZE_BATTLE, "inv": "1", **fields}
    r = client.post(path, data=data)
    assert r.status_code == 200, r.text[:400]
    assert "data-job-id" in r.text
    job_id = r.text.split('data-job-id="')[1].split('"')[0]
    _await_job(client, job_id)
    return client.get(f"/partials/job/{job_id}").text, seen


def test_lookahead_submits_polls_and_renders_its_per_action_value_delta(client):
    """The port of the TUI's `L`: per legal action, V(s′) on the realized dice, ΔV against the line
    that was played, and a terminal win/loss where the action ends the battle."""
    html, seen = _run_cf(client, "/partials/job/lookahead", "lookahead", _FAKE_LOOKAHEAD,
                         seeds="4")
    assert 'data-job-status="done"' in html
    assert 'hx-trigger="every 2s"' not in html, "a finished job must stop polling"
    # The form's fields REACHED the session — an HTMX POST carries them in the body, and a
    # parameter that read the URL only would silently probe with the defaults.
    assert seen["kwargs"]["n_seeds"] == 4 and seen["kwargs"]["inv"] == 1

    for c in _FAKE_LOOKAHEAD["candidates"]:
        assert c["label"] in html
    assert "5.25" in html, "the best alternative's ΔV is the headline"
    assert "▶" in html and "★" in html, "the chosen action and the best alternative must differ"
    assert ">loss<" in html, "an action that ENDS the battle has no V — it has a winner"
    assert "common random numbers" in html, "ΔV without the CRN framing is not interpretable"


def test_better_line_renders_the_contrastive_trajectory_and_its_provenance(client):
    """The port of `B`: "turn T: you played X → better line Y", the headline ΔV / P(win), the
    principal variation ply by ply, and WHO played the interior plies."""
    html, seen = _run_cf(client, "/partials/job/better-line", "better_line", _FAKE_BETTER_LINE,
                         depth="2", beam="3", confirm_rollouts="0")
    assert seen["kwargs"]["depth"] == 2 and seen["args"][1] == 1
    assert "turn 7: you played" in html and "thunderbolt" in html
    assert "switch:swampert" in html
    assert "5.25" in html and "61.0%" in html
    assert "principal variation" in html and "ply 2" in html
    assert "depth 2 · beam 3 · top-k 4" in html
    # The interior opponent is a flagged SELF-PROXY at depth ≥ 2 — the load-bearing caveat, since
    # only the divergence ply is played by the recorded opponent.
    assert "SELF-PROXY" in html
    assert "self_model_approx" in html
    # No rollout was requested, so the page must say these are the CRITIC's numbers, not a result.
    assert "not a played-out result" in html


def test_a_replay_counterfactual_says_a_single_rollout_is_not_a_probability(client):
    """`n_rollouts == 1` is one realized-dice line. The payload says so in its own caveats, and a
    win rate of 100% printed without it is the single most misreadable number these probes emit."""
    html, seen = _run_cf(client, "/partials/job/replay-counterfactual", "replay_counterfactual",
                         _FAKE_REPLAY_ONE, action="1", rollouts="1", narrate="on")
    assert seen["args"][1:] == (1, 1) and seen["kwargs"]["n_rollouts"] == 1
    assert seen["kwargs"]["narrate"] is True
    assert "NOT a probability" in html, "the payload's own caveat never reached the page"
    assert "realized-dice line" in html
    assert "self_model_approx" in html, "which opponent played it is part of the claim"
    # narrate=True produced a play-by-play: the trajectory is rendered, collapsed.
    assert "a recovered WIN" in html
    assert "we icebeam — opp salamence fainted" in html


def test_an_unchecked_play_by_play_box_actually_turns_narration_off(client):
    """An unchecked checkbox sends NO field, so a default of "on" would keep capturing it."""
    _, seen = _run_cf(client, "/partials/job/replay-counterfactual", "replay_counterfactual",
                      _FAKE_REPLAY_ONE, action="1")
    assert seen["kwargs"]["narrate"] is False


def test_a_lookahead_result_hands_its_best_alternative_to_the_replay(client):
    """The TUI's `L` → `C` handoff: `replay_counterfactual` has no default action, so the way to
    get one is to have just looked ahead."""
    html, _ = _run_cf(client, "/partials/job/lookahead", "lookahead", _FAKE_LOOKAHEAD)
    assert "/partials/job/replay-counterfactual" in html
    assert 'name="action" value="1"' in html, "the best NON-CHOSEN alternative, pre-filled"
    assert "replay “switch:swampert” to a win/loss" in html


def test_a_replay_without_an_action_is_a_message_not_a_500(client):
    """There is no sensible default alternative, and an HTMX swap would drop a 400 on the floor."""
    _unlock(client)
    r = client.post("/partials/job/replay-counterfactual",
                    data={"battle": _ANALYZE_BATTLE, "inv": "1"})
    assert r.status_code == 200
    assert 'data-job-status="invalid"' in r.text
    assert "pick which action to substitute" in r.text


@pytest.mark.parametrize("path,method,fields", [
    ("/partials/job/lookahead", "lookahead", {}),
    ("/partials/job/better-line", "better_line", {}),
    ("/partials/job/replay-counterfactual", "replay_counterfactual", {"action": "1"}),
])
def test_a_trace_without_a_reconstruction_record_reads_as_a_message(client, path, method, fields):
    """All three need the `*_reconstruction.json` sibling that only bridge-eval traces carry — and
    the fixture, like every websocket-era run, has none. That is an ordinary state of the data."""
    _unlock(client)
    message = ("no reconstruction record next to this trace (/x/b_reconstruction.json) — "
               "lookahead needs the re-roll layer's replay data, which only bridge-eval traces carry")

    def boom(*a, **kw):
        raise FileNotFoundError(message)

    setattr(_the_session(client), method, boom)
    r = client.post(path, data={"battle": _ANALYZE_BATTLE, "inv": "1", **fields})
    job_id = r.text.split('data-job-id="')[1].split('"')[0]
    body = _await_job(client, job_id)
    assert body["status"] == "error"

    html = client.get(f"/partials/job/{job_id}")
    assert html.status_code == 200, "a probe that cannot run is not a 500"
    assert "reconstruction record" in html.text
    assert "bridge-eval traces carry" in html.text


def test_a_counterfactual_on_an_archived_run_renders_the_drift_diagnosis_whole(client):
    """These three LOAD THE CHECKPOINT, so `ArchDriftError` is their ordinary outcome on an
    archived run (79/79 measured). Its message is a multi-line diagnosis whose last useful line is
    the exact commit to re-probe from — the job box renders it whole, like `/analyze` does."""
    from main.prober.model import ArchDriftError

    _unlock(client)
    message = ("This checkpoint cannot be re-run under the current code:\n"
               "  /models/run_x/checkpoints/checkpoint_1_steps.zip\n"
               "\n"
               "  obs dim        trained 2667  ·  code builds 2669\n"
               "  · probe from the commit it was trained on:  git checkout abc123def\n"
               "\n"
               "MODEL-FREE views need none of this and work on every archived run:\n"
               "  scan · triage · turns · overview · find · falsify · calibration")

    def boom(*a, **kw):
        raise ArchDriftError(message, saved_obs_dim=2667, current_obs_dim=2669,
                             git_hash="abc123def")

    _the_session(client).better_line = boom
    r = client.post("/partials/job/better-line", data={"battle": _ANALYZE_BATTLE, "inv": "1"})
    job_id = r.text.split('data-job-id="')[1].split('"')[0]
    _await_job(client, job_id)

    html = client.get(f"/partials/job/{job_id}").text
    assert "git checkout abc123def" in html, "the one actionable line was dropped"
    assert "obs dim        trained 2667" in html
    assert 'class="err"' in html      # `.err` is white-space: pre-wrap, so the newlines survive
    assert html.count("\n", html.index("cannot be re-run"), html.index("git checkout")) >= 4


@pytest.mark.parametrize("path,fields", [
    ("/api/jobs/lookahead", {}),
    ("/api/jobs/better-line", {}),
    ("/api/jobs/replay-counterfactual", {"action": 1}),
])
def test_the_counterfactual_api_twins_submit_the_same_jobs(client, path, fields):
    """The JSON surface an agent uses. Same registry, same gate — only the transport differs."""
    _unlock(client)
    for name, payload in (("lookahead", _FAKE_LOOKAHEAD), ("better_line", _FAKE_BETTER_LINE),
                          ("replay_counterfactual", _FAKE_REPLAY_ONE)):
        setattr(_the_session(client), name, lambda *a, _p=payload, **kw: _p)
    r = client.post(path, params={"battle": _ANALYZE_BATTLE, "inv": 1, **fields})
    assert r.status_code == 202, r.text[:300]
    body = _await_job(client, r.json()["id"])
    assert body["status"] == "done"
    assert body["params"]["battle"] == _ANALYZE_BATTLE


@pytest.mark.parametrize("path,fields", [
    ("/partials/job/lookahead", {}),
    ("/partials/job/better-line", {}),
    ("/partials/job/replay-counterfactual", {"action": "1"}),
])
def test_a_counterfactual_battle_token_is_never_joined_to_a_path(client, path, fields):
    """`runs.py`'s membership rule again, one level down: an unrecognised battle id would send
    `ProbeSession._battle` off to `build_trace_tree`, which opens whatever it is handed."""
    _unlock(client)
    r = client.post(path, data={"battle": "../../../etc/passwd", "inv": "0", **fields})
    assert r.status_code == 404
    assert "passwd" not in r.text


# ---------------------------------------------------------------------------
# The species-clause reading (`exclusive_belief`)
# ---------------------------------------------------------------------------

def test_the_clause_reading_shows_the_coherent_team_and_names_only_the_DISAGREEMENTS(client):
    """The panel's job is to hand a reader a team gen3 actually allows, and to be QUIET otherwise.
    So: the hypothesis line always renders, and per-slot rows appear ONLY where the raw top-1 and
    the clause-consistent read disagree — a row per hidden slot would be visual noise on the slots
    that were never in question."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "species clause" in body
    assert "most likely hidden team consistent with the clause" in body
    # The hypothesis names both mons…
    assert "blissey" in body and "snorlax" in body
    # …and the ONE disagreeing slot is listed with both readings side by side.
    assert "raw tyranitar" in body and "clause snorlax" in body
    # The agreeing slot (2) contributes no row of its own.
    assert "raw blissey" not in body
    # The incoherence headline is stated as numbers, not adjectives.
    assert "peak expected count 1.24" in body
    assert "duplicated top-1 guess" in body


def test_the_clause_reading_says_it_is_a_READING_AID_and_never_replaces_the_raw_belief(client):
    """The model's belief is the raw marginals. A surface that quietly showed only the adjusted
    view would substitute our arithmetic for the model's state — the interpretability failure this
    panel exists to fix, one level up. Both must be on the page, and the panel must SAY which is
    which."""
    _stub_analyze(client, _FULL_ANALYSIS)
    body = _fragment(client)
    assert "reading aid" in body
    assert "does not change what the model believes" in body
    # The raw belief panel is still there — `belief_truth` is the fixture's raw form.
    assert "species belief vs the TRUE team" in body
    assert body.index("species belief vs the TRUE team") < body.index("species clause")


def test_a_COHERENT_belief_collapses_to_one_line_instead_of_a_duplicate_table(client):
    """When nothing was adjusted, drawing a second table identical to the first is worse than
    saying so — and the reader still needs to know the check RAN."""
    import copy
    a = copy.deepcopy(_FULL_ANALYSIS)
    a["exclusive_belief"] = {
        "slots": [{"slot": 2, "top": [["blissey", 0.62]], "raw_top1": "blissey",
                   "raw_top1_prob": 0.62, "adj_top1": "blissey", "adj_top1_prob": 0.62,
                   "differs": False, "total_variation": 0.0,
                   "hypothesis": "blissey", "hypothesis_differs": False}],
        "team_hypothesis": ["blissey"], "revealed": ["tyranitar"],
        "max_expected_count": 0.62, "illegal_mass": 0.0, "duplicate_top1": 0,
        "revealed_leak_max": 0.0, "converged": True, "iterations": 0, "coherent": True,
    }
    _stub_analyze(client, a)
    body = _fragment(client)
    assert "already coherent here" in body
    assert "Nothing was adjusted" in body
    assert "raw blissey" not in body, "a coherent belief must not draw per-slot disagreement rows"


def test_a_NON_CONVERGED_clause_reading_refuses_to_claim_consistency(client):
    """An unreachable constraint set means the adjusted rows do NOT satisfy the clause. Presenting
    them as if they did is the one thing this panel must never do."""
    import copy
    a = copy.deepcopy(_FULL_ANALYSIS)
    a["exclusive_belief"] = dict(a["exclusive_belief"], converged=False)
    _stub_analyze(client, a)
    body = _fragment(client)
    assert "constraint set was unreachable" in body
    assert "best-effort only" in body


def test_the_clause_panel_is_absent_on_a_belief_off_run(client):
    """Flag-gated like every other panel here: absent must mean "the head was off", never an empty
    box that reads as a broken probe."""
    _stub_analyze(client, _BARE_ANALYSIS)
    assert "species clause" not in _fragment(client)


# ---------------------------------------------------------------------------
# THE CRITIC'S CURRENCY (`gen3_prober_winprob_currency_v1`)
#
# Under `--critic winprob` the critic IS the win-prob head: V is a probability in [0,1] rather
# than a shaped return of roughly ±30. Every threshold expressed in value units therefore means
# something different, and the failure mode is SILENT — a shaped `overvalue_tau` of 5.0 exceeds
# the entire representable range of a probability gap, so `critic_overvalued` reads a confident
# 0% that is a units error wearing the costume of a finding. These pin both eras.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wp_run(tmp_path_factory):
    return fixture_run.build_winprob(str(tmp_path_factory.mktemp("proberwp")))


@pytest.fixture()
def wp_client(wp_run):
    app = create_app(wp_run, password="test-only-password")
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def test_a_run_with_no_critic_key_reads_as_SHAPED(run):
    """The archive's rule: `--critic` landed at config version 109, so a run recorded before it
    has no key and every one of them is shaped. Absence must never be read as the new era."""
    s = ProbeSession(run)
    assert s.critic_mode() == "shaped"
    cur = s.critic_currency()
    assert cur["is_probability"] is False
    assert cur["even"] == 0.0 and cur["default_overvalue_tau"] == 5.0


def test_a_winprob_run_reads_as_a_PROBABILITY_critic(wp_run):
    s = ProbeSession(wp_run)
    assert s.critic_mode() == "winprob"
    cur = s.critic_currency()
    assert cur["is_probability"] is True
    assert cur["even"] == 0.5
    assert 0.0 < cur["default_overvalue_tau"] < 1.0, \
        "a tau outside [0,1] could never fire on a probability"


def test_an_UNREADABLE_config_falls_back_to_shaped_not_to_the_new_era(tmp_path):
    """A run whose config cannot be parsed keeps the HISTORICAL behaviour. Re-scaling an old run's
    numbers because a file failed to open would corrupt a reading rather than fail it."""
    run = fixture_run.build_winprob(str(tmp_path))
    with open(os.path.join(run, "model_config.json"), "w") as f:
        f.write("{ this is not json")
    assert ProbeSession(run).critic_mode() == "shaped"


def test_triage_centres_the_V_fallback_on_the_CRITIC_not_on_zero(run, wp_run):
    """V's even-point is 0.0 only on a shaped critic. Under winprob, V=0.0 is a CERTAIN LOSS, so a
    `V > 0` winning split would call a hopeless position 'winning'."""
    assert ProbeSession(run).triage()["winning_split"]["v_even"] == 0.0
    assert ProbeSession(wp_run).triage()["winning_split"]["v_even"] == 0.5


def test_the_overvalue_cutoff_DEFAULTS_per_currency(run, wp_run):
    from main.prober.session.aggregate import _resolve_overvalue_tau
    shaped = ProbeSession(run).critic_currency()
    wp = ProbeSession(wp_run).critic_currency()
    assert _resolve_overvalue_tau(shaped, None) == (5.0, True)
    tau, defaulted = _resolve_overvalue_tau(wp, None)
    assert defaulted and tau < 1.0
    # An EXPLICIT value always wins, in both eras — a threshold sweep must not be re-scaled.
    assert _resolve_overvalue_tau(wp, 5.0) == (5.0, False)
    assert _resolve_overvalue_tau(shaped, 0.5) == (0.5, False)


def test_a_threshold_no_gap_can_reach_is_REPORTED_not_silently_zero():
    """THE DURABLE GUARD. This is the defect the whole change was written from: a cutoff larger
    than any gap the run can produce yields `critic_overvalued: 0` BY CONSTRUCTION, which reads as
    'the critic is fine'. It must announce itself as a units error instead."""
    from main.prober.session.aggregate import _unreachable_threshold_warning
    bins = [{"gap": 0.41}, {"gap": 0.07}, {"gap": -0.33}]
    wp = {"mode": "winprob", "units": "P(win)", "span": 1.0}
    warn = _unreachable_threshold_warning(5.0, wp, bins)
    assert warn and "THRESHOLD UNREACHABLE" in warn
    assert "0.41" in warn, "the warning must name the largest gap actually observed"
    # Reachable => silent. A guard that fires on healthy input is noise, and noise gets ignored.
    assert _unreachable_threshold_warning(0.083, wp, bins) is None
    # And it is not winprob-specific: a shaped tau against shaped gaps stays silent.
    shaped = {"mode": "shaped", "units": "shaped return", "span": 60.0}
    assert _unreachable_threshold_warning(5.0, shaped, [{"gap": 8.0}]) is None


def test_the_run_summary_page_NAMES_the_critics_currency(wp_client, client):
    """The one fact that decides how to read every V on every other page. A reader who does not
    learn it on the orientation page carries the shaped scale into a probability."""
    wp_body = wp_client.get("/").text
    assert "winprob" in wp_body
    assert "P(win)" in wp_body
    assert "shaped return" in client.get("/").text


def test_the_web_does_not_force_a_SHAPED_threshold_onto_a_winprob_run(wp_client):
    """The regression that mattered most: the API/form used to pass a hardcoded 5.0 / 0.0 as
    EXPLICIT arguments, which defeated the engine's per-currency defaults entirely."""
    seen = {}
    sess = _the_session(wp_client)
    sess.triage = lambda **kw: seen.update(kw) or {"categories": [], "winning_split": {}}
    wp_client.get("/api/triage")
    assert seen["v_even"] is None, "an unset v_even must reach the engine as unset"


def test_an_EMPTY_curve_never_reports_an_unreachable_threshold():
    """Absent evidence is not a units error. With no bins the largest gap is vacuously 0, and a
    naive `tau > max_gap` would brand every threshold unreachable on a run that simply captured
    nothing to measure — an alarm that fires loudest where it knows least."""
    from main.prober.session.aggregate import _unreachable_threshold_warning
    wp = {"mode": "winprob", "units": "P(win)", "span": 1.0}
    assert _unreachable_threshold_warning(5.0, wp, []) is None
    assert _unreachable_threshold_warning(5.0, wp, [{"gap": None}]) is None


# ---------------------------------------------------------------------------
# A CHROME TIMEOUT IS NEVER A SEMANTIC OUTCOME (render_integration_test._dump_dom)
#
# These live HERE, unmarked, rather than beside the browser tests they guard: the rule is about
# what a starved run MEANS, it needs no browser to check, and a guard that only runs in a 24-minute
# tier is a guard nobody sees fail. They stub the chrome call, so they cost milliseconds.
# ---------------------------------------------------------------------------


def _timeout_after(n_failures):
    """A fake `_run_chrome` that raises TimeoutExpired for the first `n_failures` calls."""
    calls = {"n": 0}

    def fake(*_a, **_k):
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise subprocess.TimeoutExpired(cmd=["chrome"], timeout=397.0)
        return "<body data-ready='1'></body>"
    return fake, calls


def _render_module():
    from main.prober.web import render_integration_test as rit
    return rit


def test_a_transient_chrome_timeout_is_RETRIED_before_anything_is_concluded(monkeypatch):
    """The cheapest explanation for one timeout is a lost scheduling race, not a wedge. Retrying
    once turns a blip into a pass instead of into a verdict about the page."""
    rit = _render_module()
    fake, calls = _timeout_after(1)          # first attempt times out, second succeeds
    monkeypatch.setattr(rit, "_run_chrome", fake)
    monkeypatch.setattr(rit, "_load_ratio", lambda: 0.1)      # idle: a blip is the likely cause
    dom = rit._dump_dom("chrome", "http://127.0.0.1:1/scan")
    assert "data-ready" in dom
    assert calls["n"] == 2, "on an IDLE box the first timeout must be retried, not reported"


def test_a_timeout_on_a_BUSY_box_is_INCONCLUSIVE_never_a_failure(monkeypatch):
    """The defect this replaces: three win-prob-era failures at load 36.8/16 cores, every one a
    timeout and not one a layout assertion — a load average reported as a rendering bug."""
    rit = _render_module()
    fake, _ = _timeout_after(99)
    monkeypatch.setattr(rit, "_run_chrome", fake)
    monkeypatch.setattr(rit, "_load_ratio", lambda: 0.81)     # the MEASURED live-trainer band
    # `Skipped` derives from BaseException, NOT Exception — `pytest.raises(Exception)` does not
    # catch it, and a test written that way SKIPS ITSELF while looking like it asserted something.
    with pytest.raises(pytest.skip.Exception) as ei:
        rit._dump_dom("chrome", "http://127.0.0.1:1/scan")
    assert "INCONCLUSIVE" in str(ei.value)


def test_a_timeout_on_a_QUIET_box_STILL_FAILS(monkeypatch):
    """THE HALF THAT MAKES THE GUARD WORTH HAVING. Bucketing every timeout as "the box was busy"
    would quietly delete this gate — a page that cannot render in 180s on an idle machine is
    broken, and must still say so."""
    rit = _render_module()
    fake, _ = _timeout_after(99)
    monkeypatch.setattr(rit, "_run_chrome", fake)
    monkeypatch.setattr(rit, "_load_ratio", lambda: 0.1)
    with pytest.raises(AssertionError, match="THE BOX WAS QUIET"):
        rit._dump_dom("chrome", "http://127.0.0.1:1/scan")


def test_quiet_is_the_RAW_load_ratio_not_the_floored_contention_factor(monkeypatch):
    """THE BUG THIS REPLACES. `cpu_contention_factor` is a SCALING metric clamped to a floor of
    1.0, so it CANNOT distinguish a half-loaded box from an idle one. Measured 2026-09-06: at load
    12.89 on 16 cores, with chrome unable to render inside 180 s, that factor read exactly 1.0 — so
    a factor-based "quiet" bar called an 80%-utilised box idle and turned every starvation timeout
    into a hard FAILURE, strictly worse than the raw TimeoutExpired it replaced."""
    rit = _render_module()
    from utils.contention import cpu_contention_factor          # noqa: F401 — the contrast IS the test
    fake, _ = _timeout_after(99)
    monkeypatch.setattr(rit, "_run_chrome", fake)
    # 12.89/16 — the load the defect was measured at. The floored factor says 1.0 ("idle"); the
    # raw ratio says 0.81, and only the raw one gets this right.
    monkeypatch.setattr(rit, "_load_ratio", lambda: 12.89 / 16)
    with pytest.raises(pytest.skip.Exception):
        rit._dump_dom("chrome", "http://127.0.0.1:1/scan")


def test_the_retry_is_QUIET_ONLY_so_a_busy_tier_does_not_double_its_slow_path(monkeypatch):
    """A two-attempt-everywhere draft turned a 45-test tier into 7 tests in 50 minutes and was
    killed mid-line by its own wrapper, destroying the pytest summary and every failure message.
    On a busy box the FIRST timeout is already the answer."""
    rit = _render_module()
    fake, calls = _timeout_after(99)
    monkeypatch.setattr(rit, "_run_chrome", fake)
    monkeypatch.setattr(rit, "_load_ratio", lambda: 0.81)
    with pytest.raises(pytest.skip.Exception):
        rit._dump_dom("chrome", "http://127.0.0.1:1/scan")
    assert calls["n"] == 1, f"a busy box must not retry; chrome ran {calls['n']}x"


def test_a_healthy_render_passes_straight_through_the_retry_wrapper(monkeypatch):
    """The SUCCESS path, pinned without a browser.

    `_dump_dom` was split into a retry/bucket wrapper plus `_run_chrome` (the original chrome
    invocation, extracted byte-for-byte). The risk of that refactor is not the timeout branch the
    other tests cover — it is that the ordinary path grows a wrapper that drops, mangles or
    re-attempts a perfectly good render. So: one call, returned verbatim, no retry.

    This is deliberately a UNIT test. The end-to-end version needs headless chrome and a quiet box,
    and this machine carries a trainer — three attempts to confirm it live were killed by wrapper
    limits at loads from 6 to 25 on 16 cores. A stubbed check that always runs beats an
    end-to-end one that never completes.
    """
    rit = _render_module()
    calls = {"n": 0}

    def ok(*_a, **_k):
        calls["n"] += 1
        return "<body data-ready='1' data-charts='2'></body>"

    monkeypatch.setattr(rit, "_run_chrome", ok)
    for ratio in (0.1, 0.9):           # idle and busy alike — success is success
        calls["n"] = 0
        monkeypatch.setattr(rit, "_load_ratio", lambda r=ratio: r)
        dom = rit._dump_dom("chrome", "http://127.0.0.1:1/")
        assert dom == "<body data-ready='1' data-charts='2'></body>"
        assert calls["n"] == 1, "a successful render must not be retried"
