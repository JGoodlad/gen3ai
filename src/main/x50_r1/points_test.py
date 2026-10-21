"""X50 rung 1: the viewer's decision points on a public log (pure text; no binary)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.live.replay_scan import battle_lines
from main.x50_r1 import spectate as S
from main.x50_r1.points import FORCED, TURN, PointError, decision_points

DATA = Path(__file__).parent / "testdata"


def _chunked(*chunks):
    out = []
    for c in chunks:
        out += ["|", "|t:|1"] + list(c)
    return out


def test_turn_points_and_a_double_ko_forced_point_at_its_chunk_start():
    lines = ["|player|p1|a||", "|player|p2|b||", "|start", "|switch|p1a: X|X|100/100",
             "|switch|p2a: Y|Y|100/100", "|turn|1"]
    lines += _chunked(["|move|p1a: X|Explosion|p2a: Y", "|-damage|p2a: Y|0 fnt", "|faint|p2a: Y",
                       "|faint|p1a: X"])
    lines += _chunked(["|switch|p2a: Z|Z|100/100", "|switch|p1a: W|W|100/100", "|upkeep", "|turn|2"])
    pts = decision_points(lines, "p1")
    chunk2 = lines.index("|switch|p2a: Z|Z|100/100") - 2       # the bare "|" before the chunk's |t:|
    assert lines[chunk2] == "|" and lines[chunk2 + 1].startswith("|t:|")
    assert pts == [(6, TURN), (chunk2, FORCED), (len(lines), TURN)]
    # the opponent's forced switch is a `wait` for the viewer: p2 gets the same forced point, p1's is its own
    assert decision_points(lines, "p2") == pts


def test_a_switch_that_continues_after_a_pursuit_ko_is_not_a_decision():
    lines = ["|turn|4"] + _chunked(["|-activate|p1a: G|move: Pursuit", "|move|p2a: T|Pursuit|p1a: G|[from] Pursuit",
                                    "|-damage|p1a: G|0 fnt", "|faint|p1a: G", "|switch|p1a: D|D|100/100",
                                    "|upkeep", "|turn|5"])
    assert decision_points(lines, "p1") == [(1, TURN), (len(lines), TURN)]


def test_a_baton_pass_opens_a_forced_point_and_a_failed_one_does_not():
    bp = ["|turn|1"] + _chunked(["|move|p1a: J|Baton Pass|p1a: J"]) + _chunked(["|switch|p1a: G|G|50/100|[from] Baton Pass",
                                                                               "|upkeep", "|turn|2"])
    pts = decision_points(bp, "p1")
    assert pts[1][1] == FORCED and bp[pts[1][0]] == "|" and bp[pts[1][0] + 2].startswith("|switch|p1a: G")
    failed = ["|turn|1"] + _chunked(["|move|p1a: J|Baton Pass|p1a: J", "|-fail|p1a: J", "|upkeep", "|turn|2"])
    failed += _chunked(["|switch|p1a: G|G|50/100", "|upkeep", "|turn|3"])
    assert [k for _, k in decision_points(failed, "p1")] == [TURN, TURN, TURN]


def test_a_log_with_no_chunk_marks_is_refused():
    with pytest.raises(PointError):
        decision_points(["|turn|1", "|faint|p1a: X", "|switch|p1a: W|W|100/100", "|turn|2"], "p1")


def test_the_points_equal_the_trainees_real_requests_on_one_of_our_battles():
    """Our own battle carries both seats' real streams: the rule on the PUBLIC view must land exactly where the
    trainee was sent its non-wait requests (this battle has turn, faint and Baton Pass decisions, and `wait`s)."""
    meta = json.loads((DATA / "trace_loss_s0_002_summary.json").read_text())["meta"]["trace_source"]
    _, p1 = S.read_stream(DATA / "trace_loss_s0_002.p1.jsonl.gz")
    _, p2 = S.read_stream(DATA / "trace_loss_s0_002.p2.jsonl.gz")
    p1, p2 = battle_lines("\n".join(p1)), battle_lines("\n".join(p2))
    seat = meta["trainee_side"]
    own = p1 if seat == "p1" else p2
    view = S.spectator_view(p1, p2)
    real = [(view.public_pos(seat, q), f) for q, f in S.request_points(own)]
    pts = decision_points(view.lines, seat)
    assert sum(f for _, f in real) >= 3 and any("Baton Pass" in ln for ln in view.lines)
    assert [(p, k == FORCED) for p, k in pts] == real
    # the public view holds no request and no seat's private HP
    assert not any(ln.startswith("|request|") for ln in view.lines)
    hp = [ln.split("|")[4 if ln.startswith("|switch|") else 3].split()[0] for ln in view.lines
          if ln.startswith(("|-damage|", "|-heal|", "|switch|"))]
    assert hp and all(h == "0" or h.endswith("/100") for h in hp)
