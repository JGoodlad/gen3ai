"""`/game` (`web/game.py`) on the synthetic fixture run — no checkpoint, no protocol log, so this pins
the DEGRADED paths: the model-free story still renders, and every model view renders a message (the
plain arch-drift sentence for an older architecture) instead of a 500. The populated model panels on
a real battle + a current-architecture checkpoint are `main/prober/game_integration_test.py` (`sim`)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main.prober.web import fixture_run
from main.prober.web.app import create_app
from main.prober.web.game import ARCH_OLDER_TEXT, _model_error


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    # A run that RECORDS HEAD's architecture: its model views may be tried, so the gating tests below see the
    # loader / the unlock prompt. (Its checkpoint is a husk, so the load itself still fails — typed.)
    return fixture_run.build(str(tmp_path_factory.mktemp("proberweb_game")),
                             identity=fixture_run.head_identity())


@pytest.fixture()
def client(run):
    app = create_app(run, password="test-only-password")
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def _unlock(client):
    """The model views need the shared password (`web/gate.py`); the story does not."""
    r = client.post("/login", data={"password": "test-only-password", "next": "/game"},
                    follow_redirects=False)
    assert r.status_code == 303, r.text[:300]


def _battle(client):
    return client.get("/api/battles").json()[0]["short_id"]


def _sess(client):
    client.get("/api/run")
    (s,) = client.app_state.sessions.values()
    return s


def test_game_is_in_the_nav():
    from main.prober.web.app import _NAV, VIEW_QUESTIONS
    assert ("/game", "game") in _NAV
    assert any(v[0] == "/game" for v in VIEW_QUESTIONS)


def test_the_page_renders_the_story_and_a_model_slot(client):
    _unlock(client)
    b = _battle(client)
    r = client.get("/game", params={"battle": b})
    assert r.status_code == 200
    assert 'class="turnlist' in r.text and 'id="game-model"' in r.text
    assert "/partials/game/model" in r.text


def test_the_story_api_matches_the_session(client):
    b = _battle(client)
    api = client.get("/api/game/story", params={"battle": b}).json()
    sess = _sess(client)
    row = next(x for x in sess.battles() if x["short_id"] == b)
    assert api == sess.battle_story(row["id"])
    assert api["n_decisions"] == len(api["decisions"]) > 0


def test_a_run_with_no_loadable_checkpoint_renders_a_message_not_a_500(client):
    _unlock(client)
    b = _battle(client)
    r = client.get("/partials/game/model", params={"battle": b, "inv": "0"})
    assert r.status_code == 200 and 'model-unavailable' in r.text and 'data-model-state="ok"' not in r.text
    a = client.get("/partials/game/attention", params={"battle": b, "inv": "0"})
    assert a.status_code == 200 and ("warnline" in a.text or "err" in a.text) and "vega-spec" not in a.text
    assert client.get("/api/game/readout", params={"battle": b}).status_code == 400


def test_an_older_architecture_is_one_plain_sentence_with_the_diagnosis_folded(client):
    _unlock(client)
    from main.prober.model import ArchDriftError

    def drift(_battle_id):
        raise ArchDriftError("saved obs dim 2761 != current 2845\n  git checkout abc123")

    _sess(client).battle_readout = drift
    r = client.get("/partials/game/model", params={"battle": _battle(client), "inv": "0"})
    assert r.status_code == 200
    assert 'data-model-state="arch_drift"' in r.text
    assert "architecture is older than the code" in r.text
    assert "git checkout abc123" in r.text              # the diagnosis survives, folded
    e = _model_error(ArchDriftError("x"))
    assert e["kind"] == "arch_drift" and e["text"] == ARCH_OLDER_TEXT
    assert _model_error(ValueError("boom"))["kind"] == "error"


def test_an_unknown_battle_is_a_404_that_does_not_echo(client):
    r = client.get("/game", params={"battle": "step_1/../../etc"})
    assert r.status_code == 404 and "../../etc" not in r.text


def test_every_battle_surface_links_into_the_viewer_anchored_on_the_decision(client):
    b = _battle(client)
    assert "/game?" in client.get("/battle", params={"battle": b}).text
    assert "/game?" in client.get("/partials/battles").text
    scan = client.get("/partials/scan", params={"outcome": "loss"}).text
    assert "/game?" in scan and "&inv=" in scan.split("/game?", 1)[1].split('"', 1)[0]


def test_the_page_carries_the_keyboard_links_and_the_glossary(client):
    rows = client.get("/api/battles").json()
    b = next(r["short_id"] for r in rows
             if client.get("/api/game/story", params={"battle": r["short_id"]}).json()["n_decisions"] >= 2)
    first = client.get("/game", params={"battle": b, "inv": "0"}).text
    second = client.get("/game", params={"battle": b, "inv": "1"}).text
    assert 'data-nav="next"' in first and 'data-nav="prev"' not in first
    assert 'data-nav="prev"' in second
    html = second
    # a trace with no protocol log still lists every decision's turn (none is orphaned)
    story = client.get("/api/game/story", params={"battle": b}).json()
    listed = {i for t in story["turns"] for i in t["decisions"]}
    assert listed == {d["inv"] for d in story["decisions"]}
    assert "what the terms mean" in html and "OTHER_species" in html


def test_decision_index_is_clamped_not_a_500(client):
    b = _battle(client)
    assert client.get("/game", params={"battle": b, "inv": "99999"}).status_code == 200
    assert client.get("/game", params={"battle": b, "inv": "-3"}).status_code == 200


# -- ACCESS: the model views are behind the shared password (web/gate.py) -----------------------

def test_a_locked_visitor_gets_the_story_and_one_unlock_prompt_never_an_error(client):
    b = _battle(client)
    r = client.get("/game", params={"battle": b, "inv": "1"})
    assert r.status_code == 200
    assert 'class="turnlist' in r.text                                  # the model-free story renders
    assert 'data-model-state="locked"' in r.text and "Unlock to view the model" in r.text
    assert "/partials/game/model" not in r.text                         # no request that would be refused
    assert 'data-model-state="error"' not in r.text
    # the unlock link returns to THIS battle and decision
    import re
    from urllib.parse import parse_qs, unquote, urlsplit
    href = re.search(r'model-locked.*?href="(/login\?next=[^"]+)"', r.text, re.S).group(1).replace("&amp;", "&")
    nxt = unquote(parse_qs(urlsplit(href).query)["next"][0])
    assert urlsplit(nxt).path == "/game" and parse_qs(urlsplit(nxt).query)["inv"] == ["1"]
    assert parse_qs(urlsplit(nxt).query)["battle"] == [b]


def test_the_story_stays_open(client):
    assert client.get("/api/game/story", params={"battle": _battle(client)}).status_code == 200


@pytest.mark.parametrize("path", ["/api/game/readout", "/api/game/attention", "/api/analyze"])
def test_the_model_json_endpoints_are_403_with_the_way_in(client, path):
    r = client.get(path, params={"battle": _battle(client)})
    assert r.status_code == 403
    assert "/login" in r.json()["error"]


@pytest.mark.parametrize("path", ["/partials/game/model", "/partials/game/attention", "/partials/analyze"])
def test_the_model_fragments_answer_a_locked_visitor_with_a_prompt_not_a_swallowed_403(client, path):
    r = client.get(path, params={"battle": _battle(client), "inv": "0"},
                   headers={"HX-Current-URL": "http://prober.example/game?run=r&battle=b&inv=3"})
    assert r.status_code == 200
    assert 'data-model-state="locked"' in r.text and "Unlock to view the model" in r.text
    assert "next=/game%3Frun%3Dr%26battle%3Db%26inv%3D3" in r.text   # back to the page it was embedded in


def test_a_hostile_current_url_header_cannot_redirect_the_unlock(client):
    for evil in ("http://evil.example/steal?next=//x", "javascript:alert(1)", "//evil.example/game"):
        r = client.get("/partials/game/model", params={"battle": _battle(client)},
                       headers={"HX-Current-URL": evil})
        assert r.status_code == 200 and "evil" not in r.text, evil


def test_the_analyze_page_degrades_the_same_way(client):
    r = client.get("/analyze", params={"battle": _battle(client)})
    assert r.status_code == 200
    assert 'data-model-state="locked"' in r.text and "/partials/analyze" not in r.text


def test_an_open_instance_is_unlocked_without_a_password(run):
    with TestClient(create_app(run, open_access=True)) as c:
        r = c.get("/game", params={"battle": _battle(c)})
        assert 'data-model-state="locked"' not in r.text and "/partials/game/model" in r.text
        assert c.get("/partials/game/model", params={"battle": _battle(c)}).status_code == 200
