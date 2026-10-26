"""X50 rung 1: logs → labelled rows through the Rust live reader (the `live_reader` binary)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.live.reader import LiveReader, ReaderRefusal
from main.live.replay_scan import battle_lines
from main.x50_r1 import extract as E
from main.x50_r1 import spectate as S
from main.x50_r1.points import decision_points

pytestmark = pytest.mark.integration
DATA = Path(__file__).parent / "testdata"
HUMAN = DATA / "human_2642128456.log"     # a real 9-turn gen3ou replay (names anonymised), 2026-07-01


@pytest.fixture(scope="module")
def reader():
    r = LiveReader()
    yield r
    r.close()


def _human():
    lines = battle_lines(HUMAN.read_text())
    return lines, E.player_info(lines)


def _mv(m):
    from agents.gen3_data import moves
    return (E.KIND_MOVE, int(moves.get(m).num), E.SWITCH_SLOT_NONE, 0)


def _sw(sp, slot=E.SWITCH_SLOT_NONE):
    from agents.gen3_data import species
    return (E.KIND_SWITCH, 0, slot, int(species.get(sp).num))


UNK = E.UNKNOWN_LABEL


def test_golden_log_gives_exactly_the_hand_read_labels(reader):
    """Every label read off the log by hand, under the production rules (opp_intent_labels.py's table)."""
    lines, info = _human()
    want = {
        # p1 views p2: Lovely Kiss; a switch to an unseen Exeggutor; Exeggutor fainted before acting and was
        # replaced (UNKNOWN); Body Slam; Body Slam and THEN our Roar's drag (the move before the switch is their choice:
        # gen3_intent_label_move_then_switch_v1, 2026-10-26; it was UNKNOWN as "a phaze"); Thunderbolt;
        # a switch back to the REVEALED Snorlax (slot 2 of Jynx, Exeggutor, Snorlax, Starmie); a paralysis `cant`
        # (UNKNOWN); the last decision (no successor).
        "p1": [_mv("lovelykiss"), _sw("exeggutor"), UNK, _mv("bodyslam"), _mv("bodyslam"), _mv("thunderbolt"),
               _sw("snorlax", 2), UNK, UNK],
        # p2 views p1: a sleep `cant` (UNKNOWN); a switch to an unseen Dugtrio; Hidden Power (the KO) in a window that
        # closes on p2's own forced replacement — labelled since gen3_intent_label_move_then_switch_v1 (it was
        # UNKNOWN; untyped 237: a spectator never learns the type); the replacement window (no p1 action); back to the revealed
        # Suicune (slot 0); Roar; a switch to an unseen Raikou; Substitute; Thunderbolt; the last decision.
        "p2": [UNK, _sw("dugtrio"), _mv("hiddenpower"), UNK, _sw("suicune", 0), _mv("roar"), _sw("raikou"), _mv("substitute"),
               _mv("thunderbolt"), UNK],
    }
    for seat in ("p1", "p2"):
        read = E.read_spectator(reader, lines, seat, info[seat]["name"])
        lab, hp = read.labels()
        assert [tuple(int(x) for x in row) for row in lab] == want[seat], seat
        # the bare (untyped) Hidden Power flag: only p2's KO Hidden Power, labelled since the move-then-switch rule
        assert hp.tolist() == [i == 2 and seat == "p2" for i in range(len(lab))], (seat, hp.tolist())
        assert all(np.isfinite(r).all() for r in read.rows)
    # the turn-1 lead has no revealed move and no revealed bench: the declared Struggle placeholder opened it
    assert E.read_spectator(reader, lines, "p1", info["p1"]["name"]).placeholder[0]


def test_the_prefix_guard_holds_on_every_decision_of_the_golden_log(reader):
    lines, info = _human()
    for seat in ("p1", "p2"):
        pts = decision_points(lines, seat)
        read = E.read_spectator(reader, lines, seat, info[seat]["name"], points=pts)
        for k in range(len(read.rows)):
            E.prefix_guard(reader, lines, seat, info[seat]["name"], pts, k, read.rows[k])


def test_the_prefix_guard_refuses_a_row_that_saw_the_opponents_action(reader):
    """The hidden-info guard: a row built AFTER the opponent's turn-1 Lovely Kiss reached the reader (the label's
    own evidence) must not pass. Fails on revert of the guard to a no-op."""
    lines, info = _human()
    pts = decision_points(lines, "p1")
    kiss = next(i for i, ln in enumerate(lines) if "|Lovely Kiss|" in ln)
    assert pts[0][0] <= kiss < pts[1][0]
    clean = E.read_spectator(reader, lines, "p1", info["p1"]["name"], points=pts)
    leaky_pts = [(kiss + 2, pts[0][1])] + pts[1:]
    leaky = E.read_spectator(reader, lines, "p1", info["p1"]["name"], points=leaky_pts)
    assert not np.array_equal(leaky.rows[0], clean.rows[0])      # the leak is visible in the row
    with pytest.raises(E.GuardError):
        E.prefix_guard(reader, lines, "p1", info["p1"]["name"], pts, 0, leaky.rows[0])


def _trace():
    meta = json.loads((DATA / "trace_loss_s0_002_summary.json").read_text())["meta"]["trace_source"]
    _, p1 = S.read_stream(DATA / "trace_loss_s0_002.p1.jsonl.gz")
    _, p2 = S.read_stream(DATA / "trace_loss_s0_002.p2.jsonl.gz")
    p1, p2 = battle_lines("\n".join(p1)), battle_lines("\n".join(p2))
    seat = meta["trainee_side"]
    rec = json.loads((DATA / "trace_loss_s0_002_reconstruction.json").read_text())
    team = [json.loads(x.split(" ", 2)[2])["team"] for x in rec["input_log"] if x.startswith(f">player {seat}")][0]
    return seat, p1, p2, team


def test_spectator_and_full_reads_of_our_battle_agree_and_full_is_the_training_row(reader):
    """The production labeller on a SPECTATOR chain gives the labels the trainee's own chain gives, and the
    full read reproduces the rows the trainee played on (`*_states.npz`, written by the eval at the time)."""
    seat, p1, p2, team = _trace()
    own = p1 if seat == "p1" else p2
    view = S.spectator_view(p1, p2)
    name = E.player_info(view.lines)[seat]["name"]
    spec = E.read_spectator(reader, view.lines, seat, name)
    full = E.read_full(reader, own, seat, name, team)
    ls, lf = spec.labels()[0], full.labels()[0]
    assert ls.shape == lf.shape and (ls[:, 0] != E.KIND_UNKNOWN).sum() >= 10
    assert np.array_equal(ls, lf)
    states = np.load(DATA / "trace_loss_s0_002_states.npz")["obs"]
    assert np.array_equal(np.stack(full.rows), states)
    assert not np.array_equal(np.stack(spec.rows), states)     # the spectator row is public-only (design §2)


def test_a_spectator_request_is_refused_on_a_reading_that_holds_a_team(reader):
    seat, p1, p2, team = _trace()
    own = p1 if seat == "p1" else p2
    turn1 = next(i for i, ln in enumerate(own) if ln.startswith("|turn|1"))
    reader.open(seat, "x", team)
    reader.feed([ln for ln in own[:turn1 + 1] if not ln.startswith("|request|")])
    with pytest.raises(ReaderRefusal, match="spectator request"):
        reader.spectator_decide(False)


def test_an_output_under_data_is_refused(tmp_path):
    from main.x50_r1 import run
    from utils.paths import repo_path

    target = repo_path("data", "x50_r1_should_not_exist")
    assert run.main(["humans", "--dates", "2026-09-21", "--out", str(target)]) == 2
    assert not target.exists()
