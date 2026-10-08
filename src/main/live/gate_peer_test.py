"""Gate (d)'s SHADOW comparator has teeth: a one-cell row difference, a mask difference, a different action and a
different token are each reported; agreement reports nothing; the token poke-env SENT is noted on the reader."""
import numpy as np

from main.live.gate_peer import Shadow
from main.live.reader import Frame


class _Reader:
    def __init__(self, frame):
        self.frame, self.chosen, self.turn = frame, [], frame.turn

    def open(self, *a):
        pass

    def feed(self, lines):
        return self.frame if any(ln.startswith("|request|") for ln in lines) else None

    def choose(self, token):
        self.chosen.append(token)


def _shadow(tmp_path, frame, argmax=1):
    s = Shadow(tmp_path / "shadow.jsonl", model=None)
    s.reader = _Reader(frame)
    s.username = "Us"
    s._argmax = lambda row, mask: argmax  # the model forward is not under test here
    s.on_message("battle-x", ["|init|battle", "|player|p1|Us|1|", '|request|{"rqid":3}'])
    return s


def _frame():
    return Frame(side="p1", row=np.arange(8, dtype=np.float32), mask=np.array([1, 1, 0], np.int8),
                 tokens={0: "move 1", 1: "switch 2"}, turn=4, line=9, rqid=3, n=0)


def _records(tmp_path):
    import json
    return [json.loads(x) for x in (tmp_path / "shadow.jsonl").read_text().splitlines()]


def test_agreement_reports_nothing_and_notes_the_sent_token(tmp_path):
    f = _frame()
    s = _shadow(tmp_path, f)
    s.on_embed("battle-x", {"observation": f.row.copy(), "action_mask": f.mask.copy()})
    s.on_choice("battle-x", 1)
    s.on_send("battle-x", "/choose switch 2")
    (rec,) = _records(tmp_path)
    assert rec["diff"] == [] and rec["sent"] == "switch 2"
    assert s.reader.chosen == ["switch 2"]


def test_each_disagreement_is_reported(tmp_path):
    f = _frame()
    s = _shadow(tmp_path, f, argmax=0)
    row = f.row.copy()
    row[5] = np.nextafter(row[5], np.float32(99))  # ONE cell, one ulp
    s.on_embed("battle-x", {"observation": row, "action_mask": np.array([1, 0, 0], np.int8)})
    s.on_choice("battle-x", 1)
    s.on_send("battle-x", "/choose switch 3")
    (rec,) = _records(tmp_path)
    assert set(rec["diff"]) == {"row", "mask", "action", "token"}
    assert rec["row_cells"] == [5]


def test_a_stall_forfeit_closes_the_open_frame_and_the_end_is_not_a_race(tmp_path):
    """poke-env forfeits at the turn limit BEFORE it encodes: the reader's open frame is closed as a recorded
    stall forfeit (no row to compare), and the forfeit notices / win that follow are not a "race" (P4 run2: every
    one of 125 race records sat in a turn-250 battle, after the forfeit)."""
    f = _frame()
    s = _shadow(tmp_path, f)
    s.on_send("battle-x", "/forfeit")
    s.on_message("battle-x", ["|-message|Them forfeited."])
    s.on_message("battle-x", ["|", "|win|Us"])
    recs = _records(tmp_path)
    assert [r["kind"] for r in recs] == ["decision"]
    assert recs[0]["stall_forfeit"] is True and recs[0]["sent"] == "/forfeit" and recs[0]["diff"] == []


def test_a_frame_before_poke_env_answered_is_a_race(tmp_path):
    s = _shadow(tmp_path, _frame())
    s.on_message("battle-x", ["|move|p2a: X|Tackle|p1a: Y"])
    assert [r["kind"] for r in _records(tmp_path)] == ["race"]
