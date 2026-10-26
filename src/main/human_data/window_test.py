"""X59-S's GPU window runner (``window.py``, Amendment 2 M7): only exit 3 continues a student; any other exit stops
the window with no new chunk; the registered order puts every primary student before S-P0."""
from __future__ import annotations

import argparse

from main.human_data import window as W


class _P:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


def _starter(script):
    """``script[(arm, seed)]`` = the exit codes of its successive chunks."""
    calls = []

    def start(arm, seed):
        calls.append((arm, seed))
        return _P(script[(arm, seed)].pop(0))
    return start, calls


def test_only_exit_3_continues_and_0_finishes():
    assert W.next_action(0, 1) == W.DONE and W.next_action(3, 1) == W.CONTINUE
    for rc in (1, 2, 86, 87, 88, 137, 143, -9):
        assert W.next_action(rc, 1) == W.STOP
    assert W.next_action(3, W.MAX_CHUNKS) == W.STOP


def test_a_refusal_stops_the_window_and_starts_nothing_new():
    todo = [("human", 5901), ("pstar", 5901), ("selfplay", 5901), ("human", 5902)]
    start, calls = _starter({("human", 5901): [3, 3, 0], ("pstar", 5901): [1], ("selfplay", 5901): [3, 0],
                             ("human", 5902): [0]})
    res = W.run_window(todo, start, concurrency=2, log=lambda s: None, poll_s=0)
    assert res["stopped"] == {"job": ["pstar", 5901], "exit": 1, "chunks": 1} and not res["pass"]
    assert ("selfplay", 5901) not in calls and res["pending"] == [["selfplay", 5901], ["human", 5902]]


def test_a_clean_window_finishes_every_student_in_order():
    todo = W.jobs()
    assert todo[:3] == [("human", 5901), ("pstar", 5901), ("selfplay", 5901)] and len(todo) == 20
    assert all(a != "pstar0" for a, _ in todo[:15]) and all(a == "pstar0" for a, _ in todo[15:])
    start, calls = _starter({j: [3, 0] for j in todo})
    res = W.run_window(todo, start, concurrency=3, log=lambda s: None, poll_s=0)
    assert res["pass"] and len(res["done"]) == 20 and len(calls) == 40


def test_the_command_runs_each_chunk_under_the_lock_and_the_cap():
    a = argparse.Namespace(selfplay_rows="/ss", human_rows="/hr", labels="/l", labels0="/l0", python="py",
                           registration="/reg.json", dir="/d")
    c = W.command("pstar0", 5903, a)
    assert c[0].endswith("gpu_lock.sh") and c[1].endswith("mem_cap.sh") and c[2] == str(W.MEM_GB)
    assert c[c.index("--labels") + 1] == "/l0" and c[c.index("--rows") + 1] == "/hr"
    assert "--registration" in c and c[c.index("--device") + 1] == "cuda"
    assert W.command("selfplay", 5901, a)[W.command("selfplay", 5901, a).index("--rows") + 1] == "/ss"
    assert "--labels" not in W.command("human", 5901, a)
