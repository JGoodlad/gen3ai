"""THE PERSPECTIVE GUARD (designs/prober/battle_viewer_ux_2026-10-09.md §4): who could know each fact.

The battle viewer shows the board, both teams and the scouting notes under one of three perspectives —
the SPECTATOR's (public), WHAT THE MODEL SAW (public + our own team's private facts, the default) and
+ TRUTH (also the opponent's hidden facts, each MARKED). The rule is computed once per field in the
engine (`engine/perspective.py`) and applied by one macro (`fv`); this guard makes "a panel shows a
hidden fact without the marker" impossible to ship:

* the fixture's reconstruction record PLANTS facts the protocol never reveals (`fixture_story`):
  theirs (`STORY_HIDDEN`) and ours (`STORY_OURS`);
* under `model` and `public` NO planted hidden fact may appear anywhere on the page, under `public`
  no planted private fact of ours either — wherever a panel might print it;
* under `truth` every planted hidden fact appears, and only inside a `data-vis="hidden"` element
  carrying the truth marker;
* every element that renders a board fact carries `data-vis`, and the page never carries a
  visibility its perspective does not show.
Proven to fail with a field rendered bare (`{{ m.item.v }}` instead of `fv`) and with the truth view's
marker removed.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from main.prober.engine.perspective import PERSPECTIVES, perspective_board, shown
from main.prober.web import fixture_run
from main.prober.web.app import create_app
from main.prober.web.fixture_story import STORY_HIDDEN, STORY_OURS, STORY_SHORT_ID


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    run = fixture_run.build(str(tmp_path_factory.mktemp("persp")))
    with TestClient(create_app(run, password="test-only-password")) as c:
        yield c


def _html(client, view, inv):
    r = client.get("/game", params={"battle": STORY_SHORT_ID, "inv": str(inv), "view": view})
    assert r.status_code == 200
    return r.text


def _text(html):
    """The page as a reader sees it (tags stripped), so a value inside an attribute or a raw protocol
    line cannot hide or fake a hit."""
    body = html.split("<main>", 1)[1]
    return re.sub(r"<[^>]+>", " ", body)


#: A MARKED ground-truth fact as `fv` renders it: the hidden visibility, the truth class, the ◇ glyph.
_MARKED = (r'<span class="fv truth[^"]*" data-vis="hidden"[^>]*><span class="truthmark"[^>]*>◇</span>'
           r'(.*?)</span>')

# Every decision of the story battle, so each board (start-of-turn AND mid-turn replacement) is read.
_INVS = range(7)


@pytest.mark.parametrize("inv", _INVS)
def test_no_hidden_fact_reaches_the_model_or_the_spectator_view(client, inv):
    for view in ("model", "public"):
        text = _text(_html(client, view, inv))
        leaked = [v for v in STORY_HIDDEN if v in text]
        assert not leaked, f"view={view} decision {inv}: ground truth leaked unmarked: {leaked}"


@pytest.mark.parametrize("inv", _INVS)
def test_our_private_facts_never_reach_the_spectator_view(client, inv):
    html = _html(client, "public", inv)
    leaked = [v for v in STORY_OURS if v in _text(html)]
    assert not leaked, f"decision {inv}: the spectator view shows our private facts: {leaked}"
    assert 'data-vis="ours"' not in html and 'data-vis="hidden"' not in html


@pytest.mark.parametrize("inv", (0, 5))
def test_the_model_view_shows_our_private_facts_and_marks_none_hidden(client, inv):
    html = _html(client, "model", inv)
    assert 'data-vis="hidden"' not in html
    assert 'data-vis="ours"' in html, "the model's view lost our own team's private facts"
    for v in STORY_OURS:          # Blissey never came in: her set is ours alone, on our team sheet
        assert v in _text(html), f"the model's view does not show our own {v}"


@pytest.mark.parametrize("inv", _INVS)
def test_under_truth_every_planted_fact_is_shown_and_marked(client, inv):
    html = _html(client, "truth", inv)
    marked = re.findall(_MARKED, html, re.S)
    marked_text = " ".join(re.sub(r"<[^>]+>", "", m) for m in marked)
    for v in ("Choice Band", "Smeargle"):
        assert v in marked_text, f"decision {inv}: {v} is not shown as marked ground truth under + Truth"
    # …and nowhere UNMARKED: strip every marked span, the planted values must be gone from the text.
    stripped = re.sub(_MARKED, " ", html, flags=re.S)
    leaked = [v for v in STORY_HIDDEN if v in _text(stripped)]
    assert not leaked, f"decision {inv}: ground truth rendered WITHOUT the marker: {leaked}"


def test_every_rendered_board_fact_carries_its_visibility(client):
    """`fv` is the only way a board fact reaches the page; a bare value has no `data-vis`."""
    html = _html(client, "truth", 5)
    board = html.split('class="board2"', 1)[1].split("</section>\n      </div>", 1)[0]
    facts = re.findall(r'<span class="fv[^"]*" data-vis="(\w+)"', board)
    assert facts and set(facts) <= {"public", "ours", "hidden"}


def test_the_engine_rule_is_the_only_rule():
    """Unit: `shown` is the table the macro asks; every perspective shows public, only truth shows
    hidden, and the spectator never sees ours."""
    assert all(shown("public", p) for p in PERSPECTIVES)
    assert shown("ours", "model") and shown("ours", "truth") and not shown("ours", "public")
    assert shown("hidden", "truth") and not shown("hidden", "model") and not shown("hidden", "public")
    assert shown("hidden", "nonsense") is False, "an unknown perspective must read as the default"
    board = {"we": {"mons": [{"name": "Swampert", "species": "Swampert", "hp_pct": 50.0, "hp": 202,
                              "max_hp": 404, "first_turn": 0, "active": True, "moves": ["Surf"]}],
                    "team_size": 6},
             "opp": {"mons": [{"name": "Tyranitar", "species": "Tyranitar", "hp_pct": 70.0, "first_turn": 0,
                               "active": True, "moves": ["Rock Slide"], "item": None}], "team_size": 6}}
    view = perspective_board(board, our_details=[{"species": "swampert", "moves": ["surf", "protect"],
                                                  "item": "leftovers", "ability": "torrent"}],
                             opp_details=[{"species": "tyranitar", "moves": ["rockslide", "focuspunch"],
                                           "item": "choiceband", "ability": "sandstream"},
                                          {"species": "smeargle", "moves": ["spore"], "item": "laxincense"}])
    we, opp = view["we"]["mons"][0], view["opp"]["mons"]
    assert we["hp_exact"] == {"v": "202/404", "vis": "ours"} and we["species"]["vis"] == "public"
    assert [(m["v"], m["vis"]) for m in we["moves"]] == [("Surf", "public"), ("Protect", "ours")]
    assert opp[0]["item"] == {"v": "Choice Band", "vis": "hidden"}
    assert [(m["v"], m["vis"]) for m in opp[0]["moves"]] == [("Rock Slide", "public"), ("Focus Punch", "hidden")]
    assert opp[1]["species"] == {"v": "Smeargle", "vis": "hidden"} and not opp[1]["appeared"]
