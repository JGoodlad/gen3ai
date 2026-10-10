"""THE battle viewer's reading contract (designs/prober/battle_viewer_ux_2026-10-09.md), on the fixture's
STORY battle — a real gen-3 protocol + a reconstruction record (`fixture_story.py`). Each test pins one
thing the owner asked to be readable and was proven to fail with the behaviour reverted:

* the turn as ORDERED BEATS with their phases (switch → moves in order → replacements → end of turn),
  "moved first", a forced replacement linked to its decision, "never got to move";
* US vs THEM on every beat and every consequence (a side class AND the word on a chip);
* our options and their action as the SAME sorted component, the actual choice marked on both;
* the information PERSPECTIVE (spectator / what the model saw / + truth) applied per field;
* locked vs unlocked: the story everywhere, one unlock card per model slot, no refused request;
* the picker never names a different battle than the one shown.
"""

from __future__ import annotations

import html as _html
import re

import pytest
from fastapi.testclient import TestClient

from main.prober.web import fixture_run
from main.prober.web.app import create_app
from main.prober.web.fixture_story import STORY_SHORT_ID


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    # A run that RECORDS HEAD's architecture, so the model slots may be tried (locked / unlocked); the
    # checkpoint is a husk, so the model itself never loads here (`fixture_run._stamp`).
    return fixture_run.build(str(tmp_path_factory.mktemp("viewer")), identity=fixture_run.head_identity())


@pytest.fixture()
def client(run):
    with TestClient(create_app(run, password="test-only-password")) as c:
        yield c


def _unlock(c):
    r = c.post("/login", data={"password": "test-only-password", "next": "/game"}, follow_redirects=False)
    assert r.status_code == 303


def _page(c, inv, **kw):
    r = c.get("/game", params={"battle": STORY_SHORT_ID, "inv": str(inv), **kw})
    assert r.status_code == 200, r.text[:400]
    return r.text


def _beats(html):
    """(phase, side, text) per rendered beat, in page order (text as a reader sees it)."""
    out = []
    for m in re.finditer(r'<li class="beat phase-(\w+) side-(\w+)[^"]*"[^>]*>(.*?)</li>\s*(?=<li class="beat|</ol>)',
                         html, re.S):
        out.append((m.group(1), m.group(2), _html.unescape(re.sub(r"<[^>]+>", " ", m.group(3)))))
    return out


# -- the turn story -------------------------------------------------------------------------------

def test_the_viewer_opens_on_the_newest_battle_and_it_is_the_story_battle(client):
    """`/game` with no battle opens the NEWEST checkpoint's battle — here the story battle."""
    html = client.get("/game").text
    assert "vs <b>heuristic2</b>" in html and "6,000,000" in html


def test_a_turn_reads_as_ordered_beats_with_their_phases(client):
    """Turn 4: our Drill Peck KOs Celebi → THEIR forced replacement mid-turn (Spikes on the way in) →
    the end-of-turn residuals. Turn 5: their Thunderbolt KOs us first → OUR replacement (this
    decision) → residuals, and the move we chose never happened."""
    t4 = _beats(_page(client, 3))
    assert [(p, s) for p, s, _ in t4] == [("move", "we"), ("replace", "opp"), ("residual", "field")], t4
    assert "moved first" in t4[0][2] and "Drill Peck" in t4[0][2]
    assert "fainted" in t4[0][2] and "our Skarmory's Drill Peck" in t4[0][2], "the faint lost its cause"
    assert "Starmie" in t4[1][2] and "sent in after Celebi fainted" in t4[1][2]
    assert "from Spikes" in t4[1][2], "the hazard chip on the replacement is not under it"

    t5 = _page(client, 5)
    b5 = _beats(t5)
    assert [(p, s) for p, s, _ in b5][:3] == [("move", "opp"), ("replace", "we"), ("residual", "field")], b5
    assert "this decision" in b5[1][2], "the replacement beat is not linked to the decision it was"
    assert "never got to use" in t5 and "Drill Peck" in t5 and "it fainted first" in t5


def test_the_switch_phase_comes_first_and_the_second_mover_is_marked_second(client):
    t1 = _beats(_page(client, 0))
    assert [(p, s) for p, s, _ in t1] == [("switch", "we"), ("move", "opp"), ("residual", "field")]
    assert "Swampert" in t1[0][2] and "Skarmory" in t1[0][2]
    t2 = _beats(_page(client, 1))
    assert [p for p, _s, _ in t2[:2]] == ["move", "move"]
    assert "1st" in t2[0][2] and "2nd" in t2[1][2] and "moved first" in t2[0][2]
    assert "critical hit" in t2[1][2] and "not very effective" in t2[1][2], "the qualifiers left their hit"


def test_every_beat_and_every_consequence_carries_its_side(client):
    """US vs THEM consistently: the side CLASS on every beat and every effect, and the WORD on a chip
    (never colour alone)."""
    html = _page(client, 3)
    sides = [side for _p, side, _t in _beats(html)]
    assert set(sides) <= {"we", "opp", "field"} and {"we", "opp"} <= set(sides), sides
    story = html.split('class="beats"', 1)[1].split("</ol>", 1)[0]
    effs = re.findall(r'<li class="eff (\w+) side-(\w+)"', story)
    assert effs and all(side in ("we", "opp", "field") for _k, side in effs)
    assert ("damage", "opp") in effs and ("faint", "opp") in effs
    assert story.count('<span class="chip us">us</span>') >= 1
    assert story.count('<span class="chip them">them</span>') >= 2
    # a nickname never reaches the page: species only
    assert "p2a:" not in story and "p1a:" not in story


def test_damage_is_an_hp_bar_delta_with_its_cause(client):
    html = _page(client, 2)
    assert 'class="hpbar delta"' in html and 'class="hpchange loss"' in html
    assert "88% → 41%" in html and "from Sandstorm" in html and "from Leftovers" in html


# -- our options vs their action -----------------------------------------------------------------

def _dist(html, which):
    block = html.split(f'data-dist="{which}"', 1)[1].split("</ul>", 1)[0]
    rows = re.findall(r'<li class="drow\s*(marked)?\s*"[^>]*>\s*<span class="dl">(?:<span class="dmark"[^>]*>(.)</span>)?'
                      r'(.*?)</span>.*?<span class="dp">(.*?)</span>', block, re.S)
    return [(bool(m), (mark or "") + _html.unescape(re.sub(r"<[^>]+>", "", lab)).strip(), pct)
            for m, mark, lab, pct in rows]


def _num(pct):
    return 0.0 if pct.startswith("&lt;") else float(pct.rstrip("%").replace("&gt;", ""))


def test_our_options_are_sorted_and_the_choice_is_marked(client):
    rows = _dist(_page(client, 1), "ours")
    assert len(rows) == 9, rows                           # every legal option, not only the chosen one
    vals = [_num(p) for _m, _l, p in rows]
    assert vals == sorted(vals, reverse=True), rows
    assert [lab for m, lab, _ in rows if m] == ["✔Spikes"], rows
    html = _page(client, 5)
    na = html.split('class="na', 1)[1].split("</p>", 1)[0]
    assert "move0" not in na, "an empty request slot was listed as an option"


def test_the_intent_column_is_sorted_by_probability_with_what_they_did_marked(client):
    """The model's α over THEIR actions — the same component as ours, sorted, the actual one marked."""
    _unlock(client)
    sess = next(iter(client.app.state.sessions.values())) if client.app.state.sessions else None
    if sess is None:
        client.get("/api/run")
        sess = next(iter(client.app.state.sessions.values()))
    cands = [{"col": 0, "label": "Rock Slide (seen)", "term": "move seat 1", "kind": "move", "p": 0.10, "actual": False},
             {"col": 1, "label": "Earthquake (guess)", "term": "move seat 2", "kind": "move", "p": 0.55, "actual": True},
             {"col": 2, "label": "any other move", "term": "OTHER_move", "kind": "other_move", "p": 0.05, "actual": False},
             {"col": 3, "label": "switch → Celebi (guess, 40% present)", "term": "switch to slot 2",
              "kind": "switch", "p": 0.30, "actual": False}]
    fake = {"n_decisions": 7, "glossary": {"α": "a", "π": "p", "P(KO)": "k", "P(lands)": "l", "P(resolve)": "r",
                                           "P(KO first)": "f", "OTHER_move": "o", "OTHER_species": "s",
                                           "pointer score": "ps"},
            "model_resolution": {"detail": "exact"}, "intent_calibration": None, "belief_evolution": None,
            "decisions": [{"inv": i, "chosen_index": 7, "policy": [], "attention": None, "operator": None,
                           "intent": {"candidates": cands, "actual_col": 1, "actual_text": "they used Earthquake",
                                      "p_actual": 0.55, "top_hit": True}} for i in range(7)]}
    sess.battle_readout = lambda _bid: fake
    body = client.get("/partials/game/model", params={"battle": STORY_SHORT_ID, "inv": "1"}).text
    assert 'id="game-intent"' in body and 'hx-swap-oob="true"' in body, "the intent must land beside our options"
    rows = _dist(body, "intent")
    assert [lab for _m, lab, _p in rows] == ["◀Earthquake (guess)", "switch → Celebi (guess, 40% present)",
                                             "Rock Slide (seen)", "any other move"], rows
    assert [m for m, _l, _p in rows] == [True, False, False, False]


# -- locked vs unlocked ----------------------------------------------------------------------------

def test_locked_the_story_and_board_render_with_one_unlock_card_per_model_slot(client):
    html = _page(client, 3)
    assert 'class="beats"' in html and 'class="board2"' in html and 'data-dist="ours"' in html
    assert "/partials/game/model" not in html, "a locked page must make no request that would be refused"
    assert 'data-model-state="locked"' in html
    assert "to see what the model expected them to do" in html
    # the PUBLIC half of the scouting notes renders without the model
    assert 'class="card scouting"' in html and "Starmie" in html.split('class="card scouting"', 1)[1]


def test_unlocked_the_model_slot_is_requested_and_the_story_is_unchanged(client):
    locked = _page(client, 3)
    _unlock(client)
    html = _page(client, 3)
    assert "/partials/game/model?" in html and "&view=model" in html
    assert 'data-model-state="locked"' not in html
    beats = lambda h: h.split('class="beats"', 1)[1].split("</ol>", 1)[0]
    assert beats(locked) == beats(html), "the story must not depend on the lock"


# -- the picker ------------------------------------------------------------------------------------

def test_the_picker_names_the_battle_being_shown_even_outside_its_cap(tmp_path):
    """MEASURED 2026-10-09: `/game` sliced its battle list to 200 before rendering the <select>, so a
    battle outside the newest 200 showed a DIFFERENT battle as selected. The app's one picker
    (`_picker_rows`) always carries the battle shown."""
    import os
    run = fixture_run.build(str(tmp_path))
    for k in range(205):
        fixture_run._write_battle(run, 7000000 + k, "heuristic2", "win_100",
                                  [fixture_run._inv(1, "thunderbolt", 1.0)], [1.0, 2.0])
    oldest = "step_2000000/aggressive_v2/loss_001"
    assert os.path.isdir(os.path.join(run, "eval_traces", "step_2000000"))
    with TestClient(create_app(run)) as c:
        html = c.get("/game", params={"battle": oldest}).text
    sel = html.split('id="battlesel"', 1)[1].split("</select>", 1)[0]
    assert f'value="{oldest}" selected' in sel, "the picker names a different battle than the one shown"
