"""FR-2's extraction (``main.x50_r1.run traces --seat opponent``): the trace read from the OPPONENT's seat (the
sentinel as viewer and actor), its full rows on its REAL requests labelled with ITS OWN action, through the Rust live
reader and Showdown's validator. Each test fails when the seat or the label reverts to the trainee's."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from main.live.reader import LiveReader
from main.live.replay_scan import battle_lines
from main.x50_r1 import extract as E
from main.x50_r1 import run
from main.x50_r1 import spectate as S
from main.x50_r1.points import decision_points

pytestmark = pytest.mark.integration
DATA = Path(__file__).parent / "testdata"
STEM = "loss_s0_002"


@pytest.fixture(scope="module")
def trace(tmp_path_factory) -> Path:
    """The fixture trace under the names its summary records (the testdata copies carry a ``trace_`` prefix)."""
    d = tmp_path_factory.mktemp("sentinel_0")
    for p in DATA.glob(f"trace_{STEM}*"):
        shutil.copy(p, d / p.name[len("trace_"):])
    return d / f"{STEM}_summary.json"


def _opponent(sp: Path):
    ts = json.loads(sp.read_text())["meta"]["trace_source"]
    seat = {"p1": "p2", "p2": "p1"}[ts["trainee_side"]]
    _, p1 = S.read_stream(sp.parent / ts["records"]["p1"])
    _, p2 = S.read_stream(sp.parent / ts["records"]["p2"])
    p1, p2 = battle_lines("\n".join(p1)), battle_lines("\n".join(p2))
    rec = json.loads((sp.parent / ts["reconstruction"]).read_text())
    team = [json.loads(x.split(" ", 2)[2])["team"] for x in rec["input_log"] if x.startswith(f">player {seat}")][0]
    choices = [x.split(" ", 1)[1] for x in rec["input_log"] if x.startswith(f">{seat} ")]
    return ts["trainee_side"], seat, p1, p2, team, choices


def test_the_full_read_of_the_opponents_seat_labels_its_own_real_choices(trace):
    """The sentinel's own stream, its true team and REAL requests: each decision's own-action label is the choice
    the sentinel sent (the reconstruction's ``>p2 ...``), never the trainee's."""
    trainee, seat, p1, p2, team, choices = _opponent(trace)
    assert seat != trainee
    own = p1 if seat == "p1" else p2
    view = S.spectator_view(p1, p2)
    pts = decision_points(view.lines, seat)
    r = LiveReader()
    try:
        full = E.read_full(r, own, seat, E.player_info(view.lines)[seat]["name"], team)
    finally:
        r.close()
    oa = E.full_own_actions(view.lines, seat, pts, full)
    reqs = [json.loads(ln[len("|request|"):]) for ln in own
            if ln.startswith("|request|") and ln[len("|request|"):] and not json.loads(ln[len("|request|"):]).get("wait")]
    assert len(reqs) == len(choices) == len(oa) >= 20
    checked = 0
    for k, (rq, ch, got) in enumerate(zip(reqs, choices, oa)):
        if got == -1:
            continue                       # a hidden or absent choice carries no label
        verb, n = ch.split(" ")
        if verb == "move":
            want = 5 + int(n)
        else:
            nick = rq["side"]["pokemon"][int(n) - 1]["ident"].split(": ", 1)[1]
            want = [i for i, t in full.tokens[k].items() if t == f"switch {nick}"][0]
        assert got == want, (k, ch, got)
        checked += 1
    assert checked >= 0.8 * len(oa)


def test_the_trace_job_reads_the_opponents_seat_in_every_mode(trace, monkeypatch):
    monkeypatch.setattr(run, "COMPLETE", True)
    monkeypatch.setattr(run, "SEAT", "opponent")
    res = run.trace_job((0, str(trace)))
    assert "error" not in res, res.get("error")
    z = run._concat(res["parts"])
    trainee = json.loads(trace.read_text())["meta"]["trace_source"]["trainee_side"]
    opp_code = 1 if trainee == "p1" else 0
    assert set(np.unique(z["seat"]).tolist()) == {opp_code}
    assert set(np.unique(z["mode"]).tolist()) == {0, 1, 2, 3}
    c = res["counts"]
    assert c["full_own_action_labelled"] > 0 and c["c_own_action_labelled"] > 0 and c["t_own_action_labelled"] > 0
    # the full rows' labels are full_own_actions' on the sentinel's real requests, at the same decision index
    _, seat, p1, p2, team, _ = _opponent(trace)
    view = S.spectator_view(p1, p2)
    r = LiveReader()
    try:
        full = E.read_full(r, p1 if seat == "p1" else p2, seat, E.player_info(view.lines)[seat]["name"], team)
    finally:
        r.close()
    oa = E.full_own_actions(view.lines, seat, decision_points(view.lines, seat), full)
    m1 = z["mode"] == 1
    assert np.array_equal(z["own_action"][m1], np.asarray(oa, np.int8)[z["k"][m1]])
    # the trainee's seat (the default) is the other seat, with other labels
    monkeypatch.setattr(run, "SEAT", "trainee")
    zt = run._concat(run.trace_job((0, str(trace)))["parts"])
    assert set(np.unique(zt["seat"]).tolist()) == {1 - opp_code}
