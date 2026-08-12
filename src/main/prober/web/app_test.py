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

import os
import time

import pytest
from fastapi.testclient import TestClient

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
    # And it says where the model-dependent analysis actually lives, rather than just omitting it.
    assert "need the" in html and "main.prober.query analyze" in html


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
