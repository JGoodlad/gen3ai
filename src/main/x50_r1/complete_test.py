"""X50 rung 1b step 0 (F-R1-1's fix): the COMPLETED own side, through the Rust live reader and Showdown's validator.

* the synthesised requests on our own battle, given the TRUE team and the own stream's exact HP, reproduce the rows
  the trainee played on BYTE FOR BYTE, and the own-action labels are the trainee's real choices;
* C-GAME keeps every revealed own fact, passes the validator, is deterministic, explains every own action, keeps
  the production opponent labels, and holds the completed prefix guard (which refuses a leaky row);
* a refuted Choice Band is never redrawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.live.reader import LiveReader
from main.live.replay_scan import battle_lines
from main.x50_r1 import complete as C
from main.x50_r1 import extract as E
from main.x50_r1 import spectate as S
from main.x50_r1.points import decision_points

pytestmark = pytest.mark.integration
DATA = Path(__file__).parent / "testdata"
HUMAN = DATA / "human_2642128456.log"


@pytest.fixture(scope="module")
def reader():
    r = LiveReader()
    yield r
    r.close()


@pytest.fixture(scope="module")
def completer():
    return C.Completer()


def _trace():
    meta = json.loads((DATA / "trace_loss_s0_002_summary.json").read_text())["meta"]["trace_source"]
    _, p1 = S.read_stream(DATA / "trace_loss_s0_002.p1.jsonl.gz")
    _, p2 = S.read_stream(DATA / "trace_loss_s0_002.p2.jsonl.gz")
    p1, p2 = battle_lines("\n".join(p1)), battle_lines("\n".join(p2))
    seat = meta["trainee_side"]
    rec = json.loads((DATA / "trace_loss_s0_002_reconstruction.json").read_text())
    team = [json.loads(x.split(" ", 2)[2])["team"] for x in rec["input_log"] if x.startswith(f">player {seat}")][0]
    choices = [x.split(" ", 1)[1] for x in rec["input_log"] if x.startswith(f">{seat} ")]
    return seat, p1, p2, team, choices


def _exact_own_hp(view: S.SpectatorView, own: list, seat: str) -> list:
    """The public log with each OWN HP line taken from the seat's own stream (the exact figure)."""
    o = [ln for ln in own if not ln.startswith("|request|")]
    out = []
    for i, ln in enumerate(view.lines):
        p = ln.split("|")
        mine = len(p) > 2 and p[2].startswith(seat) and p[1] in ("switch", "drag", "-damage", "-heal", "-sethp")
        out.append(o[view.consumed[seat][i] - 1] if mine else ln)
    return out


def test_the_synthesised_requests_on_the_true_team_are_the_training_rows(reader):
    """THE fidelity gate of the request synthesiser: with the trainee's TRUE team and its own stream's exact HP,
    every completed row equals the row the trainee played on (`*_states.npz`), byte for byte, with the same mask.
    Fails on revert of any request rule this battle exercises (the roster, the slot swap, PP, Pressure's silent
    announce, the locked / Struggle formats, Return's power)."""
    from main.x50_r1 import run

    seat, p1, p2, team, _ = _trace()
    own = p1 if seat == "p1" else p2
    view = S.spectator_view(p1, p2)
    lines = _exact_own_hp(view, own, seat)
    cr = E.read_completed(reader, lines, seat, "rhone", team, run._real_genders(own))
    states = np.load(DATA / "trace_loss_s0_002_states.npz")["obs"]
    assert len(cr.read.rows) == len(states) >= 40
    assert np.array_equal(np.stack(cr.read.rows).view(np.uint32), states.view(np.uint32))
    full = E.read_full(reader, own, seat, "rhone", team)
    assert all(np.array_equal(a, b) for a, b in zip(full.masks, cr.read.masks))
    # ... and on the PUBLIC log (percent HP) only the HP cells move: the rewrite is the only approximation left
    pub = E.read_completed(reader, view.lines, seat, "rhone", team, run._real_genders(own))
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    n, dim = lay["parts"]["our_team"]["reshape"]
    hp_cells = {i * dim + lay["pokemon"]["hp"]["offset"] for i in range(n)}
    ev = range(lay["event_window_offset"], lay["event_window_offset"] + lay["event_window_dim"])
    moved = set().union(*(set(np.nonzero(a != b)[0].tolist()) for a, b in zip(states, pub.read.rows)))
    assert moved and moved <= hp_cells | set(ev)


def test_the_own_action_labels_are_the_trainees_real_choices(reader):
    from main.x50_r1 import run

    seat, p1, p2, team, choices = _trace()
    own = p1 if seat == "p1" else p2
    view = S.spectator_view(p1, p2)
    cr = E.read_completed(reader, view.lines, seat, "rhone", team, run._real_genders(own))
    reqs = [json.loads(ln[len("|request|"):]) for ln in own
            if ln.startswith("|request|") and ln[len("|request|"):] and not json.loads(ln[len("|request|"):]).get("wait")]
    assert len(reqs) == len(choices) == len(cr.own_action)
    checked = 0
    for k, (rq, ch, got) in enumerate(zip(reqs, choices, cr.own_action)):
        if got == -1:
            continue                       # a hidden or absent choice carries no label
        verb, n = ch.split(" ")
        if verb == "move":
            want = 5 + int(n)
        else:
            nick = rq["side"]["pokemon"][int(n) - 1]["ident"].split(": ", 1)[1]
            want = [i for i, t in cr.read.tokens[k].items() if t == f"switch {nick}"][0]
        assert got == want, (k, ch, got)
        checked += 1
    assert checked >= 35


def _human_completed(reader, completer, seat):
    lines = battle_lines(HUMAN.read_text())
    info = E.player_info(lines)
    facts = C.seat_facts(lines, seat)
    team, _ = completer.team(facts, C.seat_rng("golden", seat, 0))
    cr = E.read_completed(reader, lines, seat, info[seat]["name"], team, facts.genders)
    return lines, info, facts, team, cr


def test_c_game_keeps_every_revealed_fact_and_is_legal_and_deterministic(reader, completer):
    from agents.info_levers.packed import parse_set, species_id, toid

    for seat in ("p1", "p2"):
        lines, info, facts, team, cr = _human_completed(reader, completer, seat)
        ents = [parse_set(e) for e in team.split("]")]
        assert len(ents) == facts.team_size == 6
        by_nick = {e[0]: e for e in ents}
        for m in facts.seen:
            e = by_nick[m.nick]
            assert species_id(e) == m.sid
            moves = e[4].split(",")
            assert all(any(x == toid(r) or (r == "hiddenpower" and x.startswith("hiddenpower")) for x in moves)
                       for r in m.moves), (m, moves)
            if m.item is not None:
                assert toid(e[2]) == m.item
            if m.ability is not None:
                assert toid(e[3]) == m.ability
        assert team.split("]")[0].split("|")[0] == facts.seen[0].nick          # the lead is slot 1
        assert completer.L.team_problems([team])[0] == ()
        assert completer.team(facts, C.seat_rng("golden", seat, 0))[0] == team
        assert completer.team(facts, C.seat_rng("golden", seat, 1))[0] != team
        assert all(a != -2 for a in cr.own_action)
        assert sum(a >= 0 for a in cr.own_action) >= 5
        spec = E.read_spectator(reader, lines, seat, info[seat]["name"])
        assert np.array_equal(cr.read.labels()[0], spec.labels()[0])           # (a)'s labels are unchanged
        assert not any(cr.read.placeholder)


def test_the_completed_prefix_guard_holds_and_refuses_a_leak(reader, completer):
    seat = "p1"
    lines, info, facts, team, cr = _human_completed(reader, completer, seat)
    name = info[seat]["name"]
    pts = decision_points(lines, seat)
    locks = E.decision_locks(lines, seat, pts)
    for k in range(len(cr.read.rows)):
        E.prefix_guard_completed(reader, lines, seat, name, team, facts.genders, pts, locks, k, cr.read.rows[k])
    kiss = next(i for i, ln in enumerate(lines) if "|Lovely Kiss|" in ln)
    leaky = E.read_completed(reader, lines, seat, name, team, facts.genders,
                             points=[(kiss + 2, pts[0][1])] + pts[1:], locks=locks)
    assert not np.array_equal(leaky.read.rows[0], cr.read.rows[0])
    with pytest.raises(E.GuardError):
        E.prefix_guard_completed(reader, lines, seat, name, team, facts.genders, pts, locks, 0, leaky.read.rows[0])


def test_a_refuted_choice_band_is_never_redrawn(completer):
    facts = C.SeatFacts("p1", 1, [C.OwnSeen("Dugtrio", "dugtrio", "M", 100, ["earthquake", "rockslide"])])
    free = [completer.team(facts, C.seat_rng("cb", "p1", a))[0].split("|")[2] for a in range(8)]
    assert "choiceband" in free                                  # the Smogon prior's mode (~96 %)
    banned = [completer.team(facts, C.seat_rng("cb", "p1", a), ban_items={"Dugtrio": ["choiceband"]})[0]
              .split("|")[2] for a in range(8)]
    assert "choiceband" not in banned


def test_a_seat_with_no_smogon_prior_is_excluded_not_guessed(completer):
    facts = C.SeatFacts("p1", 1, [C.OwnSeen("Pidgey", "pidgey", "M", 100, ["gust"])])
    with pytest.raises(C.CompletionError, match="no Smogon prior"):
        completer.team(facts, C.seat_rng("x", "p1", 0))
